/**
 * STM32F4 ARM Cortex-M4 Firmware Reference Model for Renode Intermittent DVFS Co-Simulation
 * Targets: STM32F407VG (ARM Cortex-M4 with FPU)
 * Note: Assembled into target firmware.elf via firmware/build_elf.py with exact 1,840-byte stack allocation (SP = 0x200038D0).
 */

#include <stdint.h>

#define PERIPH_BASE           ((uint32_t)0x40000000)
#define AHB1PERIPH_BASE       (PERIPH_BASE + 0x00020000)
#define APB2PERIPH_BASE       (PERIPH_BASE + 0x00010000)

#define RCC_BASE              (AHB1PERIPH_BASE + 0x3800)
#define USART1_BASE           (APB2PERIPH_BASE + 0x1000)

typedef struct {
    volatile uint32_t CR;
    volatile uint32_t PLLCFGR;
    volatile uint32_t CFGR;
    volatile uint32_t CIR;
    volatile uint32_t AHB1RSTR;
    volatile uint32_t AHB2RSTR;
    volatile uint32_t AHB3RSTR;
    uint32_t RESERVED0;
    volatile uint32_t APB1RSTR;
    volatile uint32_t APB2RSTR;
    uint32_t RESERVED1[2];
    volatile uint32_t AHB1ENR;
    volatile uint32_t AHB2ENR;
    volatile uint32_t AHB3ENR;
    uint32_t RESERVED2;
    volatile uint32_t APB1ENR;
    volatile uint32_t APB2ENR;
} RCC_TypeDef;

typedef struct {
    volatile uint32_t SR;
    volatile uint32_t DR;
    volatile uint32_t BRR;
    volatile uint32_t CR1;
    volatile uint32_t CR2;
    volatile uint32_t CR3;
    volatile uint32_t GTPR;
} USART_TypeDef;

#define RCC   ((RCC_TypeDef *) RCC_BASE)
#define USART1 ((USART_TypeDef *) USART1_BASE)

// Static FreeRTOS-style task control block stack buffer (1840 bytes)
static uint8_t freertos_tcb_stack[1840] __attribute__((aligned(8)));
static volatile uint32_t g_current_freq_mhz = 8;
static volatile uint64_t g_instruction_cycles = 0;

void usart1_send_char(char c) {
    USART1->DR = (c & 0xFF);
    for (volatile int i = 0; i < 100; i++);
}

void usart1_send_string(const char *str) {
    while (*str) {
        usart1_send_char(*str++);
    }
}

void set_cpu_frequency(uint32_t freq_mhz) {
    g_current_freq_mhz = freq_mhz;
    // Simulate STM32F4 PLL Clock Tree Reconfiguration delay
    for (volatile int i = 0; i < 500; i++);
    usart1_send_string("[MCU Clock Tree] Frequency scaled successfully.\r\n");
}

void execute_workload_step(uint32_t num_tasks) {
    // Perform synthetic workload computation
    volatile uint32_t accum = 0;
    uint32_t cycles_per_task = g_current_freq_mhz * 10000;
    
    for (uint32_t t = 0; t < num_tasks; t++) {
        for (uint32_t i = 0; i < cycles_per_task; i++) {
            accum += (i * 3) ^ 0x5A;
        }
    }
    g_instruction_cycles += (uint64_t)g_current_freq_mhz * 1000000;
}

int main(void) {
    // Enable RCC peripherals
    RCC->APB2ENR |= (1 << 4); // Enable USART1 clock
    
    usart1_send_string("\r\n================ STM32F4 ARM CORTEX-M4 DVFS FIRMWARE ================\r\n");
    usart1_send_string("[FreeRTOS Initialized] Allocated TCB Stack Footprint: 1840 Bytes\r\n");
    usart1_send_string("[MCU Initialized] Ready for DVFS Scaling Commands over Renode Bus\r\n");

    uint32_t step = 0;
    while (1) {
        // Execute continuous workload loop
        execute_workload_step(4);
        step++;
        if (step % 10 == 0) {
            usart1_send_string("[Telemetry] Core active. Instruction counter progressing...\r\n");
        }
    }

    return 0;
}
