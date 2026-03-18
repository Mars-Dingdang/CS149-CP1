#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void sqrtSimd(int N, float initualGuess, float values[], float output[]) {
    static const float kThreshold = 0.00001f;
    const int simdWidth = 8;
    int i = 0;

    __m256 threshold = _mm256_set1_ps(kThreshold);
    __m256 init_guess = _mm256_set1_ps(initualGuess);
    __m256 ones = _mm256_set1_ps(1.0f);
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 three_half = _mm256_set1_ps(1.5f);
    // construct abs mask: the sign bit is the highest bit, so we set all bits to 1 except the highest bit
    __m256i abs_mask_int = _mm256_set1_epi32(0x7FFFFFFF); 
    __m256 abs_mask = _mm256_castsi256_ps(abs_mask_int);

    for(; i + simdWidth <= N; i += simdWidth) {
        __m256 x = _mm256_loadu_ps(&values[i]);
        __m256 guess = init_guess;
        __m256 g2 = _mm256_mul_ps(guess, guess);
        __m256 g2x = _mm256_mul_ps(g2, x);
        __m256 err = _mm256_and_ps(_mm256_sub_ps(g2x, ones), abs_mask); // err = abs(g2 * x - 1)
        __m256 mask = _mm256_cmp_ps(err, threshold, _CMP_GT_OS); // mask = err > threshold

        // movemask returns an integer where each bit corresponds to the result of the comparison for each element in the vector. If movemask != 0, at least one element err > threshold.
        while (_mm256_movemask_ps(mask)) {
            __m256 half_g2x = _mm256_mul_ps(half, g2x);
            __m256 factor = _mm256_sub_ps(three_half, half_g2x); // factor = 1.5 - 0.5 * g2 * x
            __m256 new_guess = _mm256_mul_ps(guess, factor); // new_guess = guess * factor
            guess = _mm256_blendv_ps(guess, new_guess, mask); // update guess only for elements where err > threshold
            g2 = _mm256_mul_ps(guess, guess);
            g2x = _mm256_mul_ps(g2, x);
            err = _mm256_and_ps(_mm256_sub_ps(g2x, ones), abs_mask); // err = abs(g2 * x - 1)
            mask = _mm256_cmp_ps(err, threshold, _CMP_GT_OS); // update mask
        }
        __m256 result = _mm256_mul_ps(x, guess); // result = x * guess
        _mm256_storeu_ps(&output[i], result);
    }
    for(; i < N; ++ i) {
        float x = values[i];
        float guess = initualGuess;
        float err = fabs(guess * guess * x - 1.f);
        while (err > kThreshold) {
            guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
            err = fabs(guess * guess * x - 1.f);
        }
        output[i] = x * guess;
    }
}