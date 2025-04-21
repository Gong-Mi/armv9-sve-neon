#define _GNU_SOURCE // Needed for CPU affinity functions and macros
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <sched.h>  // For CPU affinity functions
#include <unistd.h> // For usleep, getpid
#include <errno.h>  // For error checking

// Include SVE and NEON headers unconditionally first
#include <arm_neon.h>
#include <arm_sve.h> // Keep SVE header included even if check fails later

// --- Timing Function ---
long long get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// --- NEON Implementation ---
double calculate_pi_neon(long long N) {
    if (N < 2) return 0.0;
    float64x2_t sum_vec = vdupq_n_f64(0.0);
    float64x2_t den_vec = {1.0, 3.0};
    const float64x2_t four_vec = vdupq_n_f64(4.0);
    // const float64x2_t one_vec = vdupq_n_f64(1.0); // Marked as unused, can be removed or kept
    const float64x2_t sign_vec = {1.0, -1.0};
    long long iterations = N / 2;
    for (long long i = 0; i < iterations; ++i) {
        float64x2_t terms = vdivq_f64(sign_vec, den_vec);
        sum_vec = vaddq_f64(sum_vec, terms);
        den_vec = vaddq_f64(den_vec, four_vec);
    }
    double sum = vgetq_lane_f64(sum_vec, 0) + vgetq_lane_f64(sum_vec, 1);
    // Handle potential odd number of terms left over
    if (N % 2 != 0) {
        // The loop handled pairs 0,1; 2,3; ...; N-3, N-2
        // The last term is N-1
        long long last_index = N - 1;
        double last_den = (double)last_index * 2.0 + 1.0;
        double last_term = (last_index % 2 == 0) ? (1.0 / last_den) : (-1.0 / last_den);
        sum += last_term;
    }
    return sum * 4.0;
}

// --- SVE Implementation (Zeroing Strategy - Verified Correct on Termux) ---
double calculate_pi_sve(long long N) {
    uint64_t lanes = svcntd(); // Get SVE vector length in doubles
    if (lanes == 0) {
        fprintf(stderr, "Error: svcntd() returned 0 lanes. SVE might not be supported or enabled properly.\n");
        return NAN; // Return Not-a-Number on critical error
    }
    if (N <= 0) return 0.0;

    svfloat64_t sum_vec = svdup_n_f64(0.0);
    long long i = 0;
    svbool_t pg_all = svptrue_b64(); // Predicate true for all lanes

    while (i < N) {
        // Create predicate for active lanes in this iteration (handles tail)
        svbool_t pg_iter = svwhilelt_b64((int64_t)i, (int64_t)N);

        // Generate indices for active lanes: 0, 1, 2, ... (lanes-1)
        svuint64_t indices_base = svindex_u64(0ULL, 1ULL);
        // Add current offset i to base indices: i, i+1, i+2, ...
        svuint64_t indices = svadd_u64_z(pg_iter, indices_base, svdup_n_u64(i)); // Zeroing inactive

        // Convert indices to float (active lanes only, others zero)
        svfloat64_t f_indices = svcvt_f64_u64_z(pg_iter, indices);

        // Calculate denominators: den = f_indices * 2.0 + 1.0 (active lanes only, others zero)
        // Using Fused Multiply Add (Zeroing): adds 1.0 to (f_indices * 2.0)
        svfloat64_t den_vec = svmla_n_f64_z(pg_iter, svdup_n_f64(1.0), f_indices, 2.0); // den = 1.0 + f_indices * 2.0 (zeroing)

        // Calculate terms: term = 1.0 / den (active lanes only, others zero)
        // Using zeroing division avoids division by zero for inactive lanes
        svfloat64_t terms = svdiv_f64_z(pg_iter, svdup_n_f64(1.0), den_vec);

        // Determine sign multiplier: +1.0 if even index, -1.0 if odd index
        svuint64_t odd_mask = svand_n_u64_z(pg_iter, indices, 1ULL); // Get LSB (zeroing)
        svbool_t is_odd_pg = svcmpne_n_u64(pg_iter, odd_mask, 0ULL); // Check if LSB is non-zero (is odd)
        // Select -1.0 where odd, +1.0 where even (or inactive)
        svfloat64_t sign_multiplier = svsel_f64(is_odd_pg, svdup_n_f64(-1.0), svdup_n_f64(1.0));

        // Apply sign (active lanes only, others zero)
        svfloat64_t signed_terms = svmul_f64_z(pg_iter, terms, sign_multiplier);

        // Accumulate - unpredicated add works because inactive lanes in signed_terms are zero
        sum_vec = svadd_f64_x(pg_all, sum_vec, signed_terms);

        i += lanes; // Move to the next block of lanes
    }

    // Horizontal sum reduction across all lanes
    double sum = svaddv_f64(pg_all, sum_vec);

    return sum * 4.0;
}


int main(int argc, char *argv[]) {
    long long N = 100000000; // Default iterations
    int target_core = -1;    // Default: no specific core

    // --- Parse Command Line Arguments ---
    if (argc > 1) {
        N = atoll(argv[1]);
    }
    if (argc > 2) {
        target_core = atoi(argv[2]); // Optional target core number
    }

    if (N <= 0) {
        fprintf(stderr, "Error: N must be positive.\n"); return 1;
    }

    printf("Combined Pi Calculation Comparison (Leibniz, N = %lld) - v4 Zeroing\n", N); // Version marker

    // --- Set CPU Affinity (if target_core is specified) ---
    if (target_core >= 0) {
        // Check if running inside GitHub Actions where affinity might fail or be less meaningful
        if (getenv("GITHUB_ACTIONS") != NULL) {
             printf("Running inside GitHub Actions. Skipping CPU affinity setting.\n");
             target_core = -1; // Reset target_core to avoid confusion in output
        } else {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(target_core, &cpuset);
            pid_t pid = getpid(); // Get current process ID
            if (sched_setaffinity(pid, sizeof(cpu_set_t), &cpuset) == -1) {
                perror("sched_setaffinity failed");
                fprintf(stderr, "Warning: Could not set CPU affinity to core %d for PID %d. Running on any allowed core.\n", target_core, pid);
                target_core = -1; // Indicate failure
            } else {
                printf("Successfully set CPU affinity to core %d for PID %d.\n", target_core, pid);
            }
        }
    } else {
        printf("No specific target core specified. Running on any allowed core.\n");
    }
    printf("--------------------------------------------------\n");

    long long start_time_neon, end_time_neon;
    long long start_time_sve, end_time_sve;
    long long duration_neon; // Declaration moved before use
    long long duration_sve; // <<< DECLARED HERE >>>
    double pi_neon_res, pi_sve_res;
    double error_neon, error_sve;
    const double pi_ref = M_PI;

    // --- Run NEON version ---
    printf("Executing NEON version...\n");
    start_time_neon = get_time_ns();
    pi_neon_res = calculate_pi_neon(N);
    end_time_neon = get_time_ns();
    duration_neon = end_time_neon - start_time_neon; // Calculate duration here
    error_neon = fabs(pi_neon_res - pi_ref);
    printf("  NEON Result: %.15f (Error: %.15f)\n", pi_neon_res, error_neon);
    printf("  NEON Time:   %lld ns (%f seconds)\n", duration_neon, (double)duration_neon / 1e9);
    printf("\n");

    // --- Run SVE version ---
    printf("Executing SVE version (Zeroing Strategy)...\n");
    start_time_sve = get_time_ns();
    pi_sve_res = calculate_pi_sve(N); // Call the SVE function
    end_time_sve = get_time_ns();

    if (isnan(pi_sve_res)) {
        printf("  SVE Calculation failed (returned NAN).\n");
        duration_sve = -1; // Indicate failure
    } else {
        error_sve = fabs(pi_sve_res - pi_ref);
        printf("  SVE Result:  %.15f (Error: %.15f)\n", pi_sve_res, error_sve);
        duration_sve = end_time_sve - start_time_sve; // Calculate duration here
        printf("  SVE Time:    %lld ns (%f seconds)\n", duration_sve, (double)duration_sve / 1e9);
    }
     printf("\n");


    // --- Final Comparison ---
    printf("---------------- Comparison ------------------\n");
    printf("NEON Duration: %lld ns\n", duration_neon);
    if (duration_sve >= 0) { // Only print SVE duration if calculation succeeded
        printf("SVE Duration:  %lld ns\n", duration_sve);
        if (duration_neon > 0 && duration_sve > 0) {
            double diff = (double)duration_neon - (double)duration_sve;
            double percentage_diff = (diff / (double)duration_neon) * 100.0; // Renamed for clarity
            double ratio = (double)duration_neon / (double)duration_sve; // Calculate ratio only once

            printf("Difference (NEON - SVE): %.2f ns\n", diff);
            if (diff > 0) {
                 printf("SVE was faster by %.2f%%\n", percentage_diff);
            } else if (diff < 0) {
                 // Calculate percentage relative to SVE time if NEON is faster
                 percentage_diff = (-diff / (double)duration_sve) * 100.0;
                 printf("NEON was faster by %.2f%%\n", percentage_diff);
            } else {
                 printf("NEON and SVE time was equal.\n");
            }
            printf("Ratio (NEON time / SVE time): %.3f\n", ratio);
        } else {
             printf("Comparison skipped (one or both durations were zero or negative).\n");
        }
    } else {
        printf("SVE Calculation failed, comparison skipped.\n");
    }
    // Adjusted target core output check
    const char* gh_actions_env = getenv("GITHUB_ACTIONS");
    printf("Target Core: %s\n", (target_core >= 0 && gh_actions_env == NULL && argc > 2) ? argv[2] : "Any/Not Set");
    printf("Reference Pi (M_PI): %.15f\n", pi_ref);
    printf("----------------------------------------------\n");


    return 0;
}