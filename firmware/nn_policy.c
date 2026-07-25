#include <math.h>
#include "nn_policy.h"

int predict_action_on_chip(const float obs[5]) {
    float h1[64];
    float h2[64];

    // Layer 1: Dense 5 -> 64 with Tanh activation
    for (int i = 0; i < 64; i++) {
        float sum = B1_BIASES[i];
        for (int j = 0; j < 5; j++) {
            sum += W1_WEIGHTS[i][j] * obs[j];
        }
        h1[i] = tanhf(sum);
    }

    // Layer 2: Dense 64 -> 64 with Tanh activation
    for (int i = 0; i < 64; i++) {
        float sum = B2_BIASES[i];
        for (int j = 0; j < 64; j++) {
            sum += W2_WEIGHTS[i][j] * h1[j];
        }
        h2[i] = tanhf(sum);
    }

    // Output Head: Dense 64 -> 4 Action Logits (Argmax for deterministic policy)
    int best_action = 0;
    float max_logit = -1e9f;
    for (int i = 0; i < 4; i++) {
        float sum = B3_BIASES[i];
        for (int j = 0; j < 64; j++) {
            sum += W3_WEIGHTS[i][j] * h2[j];
        }
        if (sum > max_logit) {
            max_logit = sum;
            best_action = i;
        }
    }

    return best_action;
}
