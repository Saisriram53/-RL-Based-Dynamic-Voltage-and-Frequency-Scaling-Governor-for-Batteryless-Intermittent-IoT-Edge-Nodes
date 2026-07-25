#include <stdint.h>

extern uint32_t _sidata;
extern uint32_t _sdata;
extern uint32_t _edata;
extern uint32_t _sbss;
extern uint32_t _ebss;

extern int main(void);

void Reset_Handler(void);
void Default_Handler(void) { while (1); }

// Vector Table definition
__attribute__((section(".vector_table")))
const uint32_t vector_table[] = {
    0x20004000,                  // Initial Stack Pointer (MSP)
    (uint32_t)&Reset_Handler,   // Reset Handler
    (uint32_t)&Default_Handler, // NMI Handler
    (uint32_t)&Default_Handler  // HardFault Handler
};

void Reset_Handler(void) {
    // 1. Enable FPU Coprocessor CP10 and CP11
    volatile uint32_t *CPACR = (volatile uint32_t *)0xE000ED88;
    *CPACR |= ((3UL << 20) | (3UL << 22)); // Full access to CP10 & CP11 FPU
    __asm__ volatile ("dsb \n isb");

    // 2. Copy .data section from Flash to SRAM
    uint32_t *src = &_sidata;
    uint32_t *dst = &_sdata;
    while (dst < &_edata) {
        *dst++ = *src++;
    }

    // 3. Zero fill .bss section in SRAM
    dst = &_sbss;
    while (dst < &_ebss) {
        *dst++ = 0;
    }

    // 4. Jump to C main()
    main();

    while (1);
}
