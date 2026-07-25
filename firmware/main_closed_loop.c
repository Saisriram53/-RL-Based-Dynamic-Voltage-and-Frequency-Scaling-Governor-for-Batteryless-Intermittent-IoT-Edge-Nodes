#include <stdint.h>
#include "nn_policy.h"

/*
 * Memory-Mapped I/O (MMIO) Closed-Loop Mailbox Buffer in SRAM
 * Base Address: 0x20001000
 *
 * 0x20001000 - 0x20001013: float obs[5] (V_terminal, Q_len, P_harvest, dP_harvest, a_prev/3.0)
 * 0x20001014: uint32_t action_out (0=8MHz, 1=16MHz, 2=48MHz, 3=80MHz)
 * 0x20001018: uint32_t status_flag (1=New Input Ready, 2=Inference Complete)
 */

#define MMIO_BASE               0x20001000UL
#define MMIO_OBS_IN             ((volatile float *)(MMIO_BASE))
#define MMIO_ACTION_OUT         ((volatile uint32_t *)(MMIO_BASE + 0x14))
#define MMIO_STATUS_FLAG        ((volatile uint32_t *)(MMIO_BASE + 0x18))

int main(void) {
    // Initialize status flag
    *MMIO_STATUS_FLAG = 0;
    *MMIO_ACTION_OUT = 0;

    while (1) {
        // Poll for new input observation written by Python co-simulation script
        if (*MMIO_STATUS_FLAG == 1) {
            float obs[5];
            obs[0] = MMIO_OBS_IN[0];
            obs[1] = MMIO_OBS_IN[1];
            obs[2] = MMIO_OBS_IN[2];
            obs[3] = MMIO_OBS_IN[3];
            obs[4] = MMIO_OBS_IN[4];

            // Run ON-CHIP PPO Neural Network Inference in C
            int action = predict_action_on_chip(obs);

            // Write output action and acknowledge status flag
            *MMIO_ACTION_OUT = (uint32_t)action;
            *MMIO_STATUS_FLAG = 2; // Signal Python co-sim harness that inference is done
        }
    }

    return 0;
}
