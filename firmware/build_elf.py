import struct
import os

def create_arm_cortex_m4_elf(output_filepath):
    """
    Generates a valid 32-bit Little-Endian ARM ELF binary (EM_ARM)
    targeting ARM Cortex-M4 memory layout (Flash at 0x08000000, SRAM at 0x20000000).
    Includes Vector Table, Reset Handler, and USART1 UART telemetry Thumb-2 instructions.
    """
    # 1. Vector Table (at 0x08000000)
    # Entry 0: Initial Stack Pointer (MSP) = 0x20004000 (SRAM Top)
    # Entry 1: Reset Handler Address      = 0x08000009 (0x08000008 | 1 for Thumb bit)
    msp_init = 0x20004000
    reset_vector = 0x08000009
    
    vector_table = struct.pack('<II', msp_init, reset_vector)
    
    # 2. Thumb-2 Cortex-M4 Instruction Bytes (Reset Handler code at 0x08000008)
    # Assembly instructions (Little-Endian 16-bit Thumb instructions):
    #   0x08000008: 2041      movs r0, #65        ; 'A'
    #   0x0800000A: 4902      ldr  r1, [pc, #8]   ; Load USART1 DR address
    #   0x0800000C: 6008      str  r0, [r1, #0]   ; Write to USART1 DR
    #   0x0800000E: e7fd      b    0x0800000E     ; Infinite loop (b .)
    #   0x08000010: 40011004  .word 0x40011004    ; Address of USART1 DR
    code_instructions = bytes([
        0x41, 0x20,  # movs r0, #65
        0x02, 0x49,  # ldr  r1, [pc, #8]
        0x08, 0x60,  # str  r0, [r1, #0]
        [0xfd, 0xe7][0], [0xfd, 0xe7][1], # b .
        0x00, 0x00,  # nop alignment
        0x04, 0x10, 0x01, 0x40 # 0x40011004 (USART1 DR address)
    ])
    
    # Padding to make text section aligned
    text_payload = vector_table + code_instructions
    text_size = len(text_payload)
    
    # 3. Construct ELF Header (52 bytes)
    # e_ident: \x7fELF, 32-bit (1), Little Endian (1), Version 1, System V ABI (0)
    e_ident = b'\x7fELF\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    e_type = 2          # ET_EXEC (Executable)
    e_machine = 40      # EM_ARM
    e_version = 1
    e_entry = reset_vector  # Entry Point (0x08000009)
    e_phoff = 52        # Program header offset (immediately after ELF header)
    e_shoff = 0         # No section headers needed for Renode LoadELF
    e_flags = 0x05000000 # EF_ARM_EABI_VER5
    e_ehsize = 52       # ELF header size
    e_phentsize = 32    # Program header entry size
    e_phnum = 1         # 1 Program header (LOAD segment)
    e_shentsize = 0
    e_shnum = 0
    e_shstrndx = 0

    elf_header = struct.pack(
        '<16sHHIIIIIHHHHHH',
        e_ident, e_type, e_machine, e_version, e_entry,
        e_phoff, e_shoff, e_flags, e_ehsize,
        e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx
    )

    # 4. Construct Program Header (32 bytes)
    # Segment 0: PT_LOAD (1) at Flash Address 0x08000000
    p_type = 1          # PT_LOAD
    p_offset = 52 + 32  # File offset where text payload begins (84 bytes)
    p_vaddr = 0x08000000 # Flash Base Address
    p_paddr = 0x08000000 # Physical Flash Address
    p_filesz = text_size
    p_memsz = text_size
    p_flags = 5         # PF_R | PF_X (Read + Execute)
    p_align = 4

    prog_header = struct.pack(
        '<IIIIIIII',
        p_type, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_flags, p_align
    )

    full_elf = elf_header + prog_header + text_payload

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, 'wb') as f:
        f.write(full_elf)

    print(f"Successfully generated ARM Cortex-M4 ELF binary ({len(full_elf)} bytes) at: {output_filepath}")

if __name__ == "__main__":
    elf_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "firmware.elf")
    create_arm_cortex_m4_elf(elf_file)
