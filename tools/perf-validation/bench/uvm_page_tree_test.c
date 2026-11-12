/*
 * uvm_page_tree_test: run nvidia-uvm's built-in page tree test, which includes
 * test_tlb_batch_coalesce() for the TLB batch range merging.
 *
 * The test runs inside the kernel against fake GPUs and fake TLB invalidate
 * HALs, so it checks the exact invalidates a batch emits without needing a
 * particular GPU. nvidia-uvm must be loaded with built-in tests enabled:
 *
 *   sudo modprobe -r nvidia_uvm
 *   sudo modprobe nvidia_uvm uvm_enable_builtin_tests=1
 *   ./uvm_page_tree_test
 *
 * Failures are logged to the kernel log (dmesg) by the test itself.
 *
 * Build: make uvm_page_tree_test     (plain C, no CUDA needed)
 */

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

// From kernel-open/nvidia-uvm/uvm_linux_ioctl.h
#define UVM_INITIALIZE 0x30000001

typedef struct
{
    uint64_t flags __attribute__((aligned(8)));  // IN
    uint32_t rmStatus;                           // OUT
} UVM_INITIALIZE_PARAMS;

// From kernel-open/nvidia-uvm/uvm_test_ioctl.h:
// UVM_TEST_PAGE_TREE = UVM_TEST_IOCTL_BASE(10) = UVM_IOCTL_BASE(200 + 10)
#define UVM_TEST_PAGE_TREE 210

typedef struct
{
    uint32_t rmStatus;  // OUT
} UVM_TEST_PAGE_TREE_PARAMS;

int main(void)
{
    UVM_INITIALIZE_PARAMS init;
    UVM_TEST_PAGE_TREE_PARAMS test;
    int fd;

    fd = open("/dev/nvidia-uvm", O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "open /dev/nvidia-uvm: %s\n", strerror(errno));
        return 1;
    }

    memset(&init, 0, sizeof(init));
    if (ioctl(fd, UVM_INITIALIZE, &init) != 0 || init.rmStatus != 0) {
        fprintf(stderr, "UVM_INITIALIZE failed: errno %d, rmStatus 0x%x\n", errno, init.rmStatus);
        return 1;
    }

    memset(&test, 0, sizeof(test));
    if (ioctl(fd, UVM_TEST_PAGE_TREE, &test) != 0) {
        fprintf(stderr,
                "UVM_TEST_PAGE_TREE ioctl failed: %s\n"
                "Is nvidia_uvm loaded with uvm_enable_builtin_tests=1?\n",
                strerror(errno));
        return 1;
    }

    if (test.rmStatus != 0) {
        fprintf(stderr, "page tree test FAILED: rmStatus 0x%x (see dmesg)\n", test.rmStatus);
        return 1;
    }

    printf("page tree test passed (includes test_tlb_batch_coalesce)\n");
    close(fd);
    return 0;
}
