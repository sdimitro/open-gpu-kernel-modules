/*******************************************************************************
    Copyright (c) 2016 NVIDIA Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/

#include "uvm_tlb_batch.h"
#include "uvm_hal.h"
#include "uvm_gpu.h"
#include "uvm_linux.h"

// Merge each queued range into the previous one when they overlap or touch and
// use the same page sizes. Without merging, a run of N adjacent pages queued
// one page at a time uses N entries, and anything beyond
// UVM_TLB_BATCH_MAX_ENTRIES falls back to invalidating every TLB entry of the
// page tree. Writable at runtime so the two behaviors can be compared on the
// same build:
//   echo 0 > /sys/module/nvidia_uvm/parameters/uvm_perf_tlb_batch_coalesce
static int uvm_perf_tlb_batch_coalesce = 1;
module_param(uvm_perf_tlb_batch_coalesce, int, S_IRUGO|S_IWUSR);
MODULE_PARM_DESC(uvm_perf_tlb_batch_coalesce,
                 "Merge adjacent TLB invalidate ranges within a batch (1, default) or queue every range separately (0)");

bool uvm_tlb_batch_coalesce_enabled(void)
{
    return READ_ONCE(uvm_perf_tlb_batch_coalesce) != 0;
}

void uvm_tlb_batch_begin(uvm_page_tree_t *tree, uvm_tlb_batch_t *batch)
{
    memset(batch, 0, sizeof(*batch));
    batch->tree = tree;
}

static NvU64 smallest_page_size(NvU64 page_sizes)
{
    UVM_ASSERT(page_sizes != 0);

    return 1ULL << __ffs(page_sizes);
}

static NvU64 biggest_page_size(NvU64 page_sizes)
{
    UVM_ASSERT(page_sizes != 0);

    return 1ULL << __fls(page_sizes);
}

static void tlb_batch_flush_invalidate_per_va(uvm_tlb_batch_t *batch, uvm_push_t *push)
{
    uvm_page_tree_t *tree = batch->tree;
    uvm_gpu_phys_address_t pdb_addr = uvm_page_tree_pdb_address(tree);
    uvm_membar_t membar = UVM_MEMBAR_NONE;
    NvU32 i;

    for (i = 0; i < batch->count; ++i) {
        uvm_tlb_batch_range_t *entry = &batch->ranges[i];
        NvU64 min_page_size = smallest_page_size(entry->page_sizes);
        NvU64 max_page_size = biggest_page_size(entry->page_sizes);

        // Use the depth of the max page size as it's the broadest
        NvU32 depth = tree->hal->page_table_depth(max_page_size);

        UVM_ASSERT(hweight64(entry->page_sizes) > 0);

        // Do the required membar only after the last invalidate
        if (i == batch->count - 1)
            membar = batch->membar;

        // Use the min page size for the targeted VA invalidate as each page
        // needs to be invalidated separately.
        tree->gpu->parent->host_hal->tlb_invalidate_va(push,
                                                       pdb_addr,
                                                       depth,
                                                       entry->start,
                                                       entry->size,
                                                       min_page_size,
                                                       membar);
    }
}

static void tlb_batch_flush_invalidate_all(uvm_tlb_batch_t *batch, uvm_push_t *push)
{
    uvm_page_tree_t *tree = batch->tree;
    uvm_gpu_t *gpu = tree->gpu;
    NvU32 page_table_depth = tree->hal->page_table_depth(batch->biggest_page_size);

    gpu->parent->host_hal->tlb_invalidate_all(push, uvm_page_tree_pdb_address(tree), page_table_depth, batch->membar);
}

static bool tlb_batch_should_invalidate_all(uvm_tlb_batch_t *batch)
{
    if (batch->count > UVM_TLB_BATCH_MAX_ENTRIES)
        return true;

    return batch->total_ranges > batch->tree->gpu->parent->tlb_batch.max_ranges;
}

void uvm_tlb_batch_end(uvm_tlb_batch_t *batch, uvm_push_t *push, uvm_membar_t tlb_membar)
{
    uvm_parent_gpu_t *parent_gpu;

    if (batch->count == 0)
        return;

    parent_gpu = batch->tree->gpu->parent;

    batch->membar = uvm_membar_max(tlb_membar, batch->membar);

    if (tlb_batch_should_invalidate_all(batch)) {
        tlb_batch_flush_invalidate_all(batch, push);
        atomic64_inc(&parent_gpu->stats.tlb_batch.invalidate_all);
    }
    else {
        tlb_batch_flush_invalidate_per_va(batch, push);
        atomic64_inc(&parent_gpu->stats.tlb_batch.targeted);
        atomic64_add(batch->count, &parent_gpu->stats.tlb_batch.targeted_ranges);
    }

    if (batch->coalesced)
        atomic64_add(batch->coalesced, &parent_gpu->stats.tlb_batch.coalesced);
}

// Try to merge [start, start + size) into the most recently queued range.
//
// Only ranges with identical page sizes masks are merged, so the merged entry
// keeps the same TLB depth and invalidate granularity, and it covers exactly
// the union of the two ranges. Ranges are only merged when they overlap or
// touch, so no address outside the requested ranges is added by the merge
// itself. (Turing+ Host rounds a ranged invalidate up to an aligned
// power-of-two span, which can cover more than requested; invalidating extra
// entries is always safe.)
static bool tlb_batch_try_coalesce(uvm_tlb_batch_t *batch, NvU64 start, NvU64 size, NvU64 page_sizes)
{
    uvm_tlb_batch_range_t *last;
    NvU64 end;
    NvU64 last_end;
    NvU64 new_start;
    NvU64 new_end;

    if (!uvm_tlb_batch_coalesce_enabled())
        return false;

    // Nothing queued to merge with, or the batch already fell back to
    // invalidate all. In the latter case ranges[] no longer tracks the queued
    // ranges, so leave the counting to the regular path.
    if (batch->count == 0 || tlb_batch_should_invalidate_all(batch))
        return false;

    last = &batch->ranges[batch->count - 1];
    if (last->page_sizes != page_sizes)
        return false;

    end = start + size;
    last_end = last->start + last->size;

    // Disjoint and not adjacent
    if (end < last->start || start > last_end)
        return false;

    new_start = min(last->start, start);
    new_end = max(last_end, end);

    last->start = new_start;
    last->size = new_end - new_start;

    ++batch->coalesced;

    return true;
}

void uvm_tlb_batch_invalidate(uvm_tlb_batch_t *batch, NvU64 start, NvU64 size, NvU64 page_sizes, uvm_membar_t tlb_membar)
{
    uvm_tlb_batch_range_t *new_entry;

    batch->membar = uvm_membar_max(tlb_membar, batch->membar);

    if (tlb_batch_try_coalesce(batch, start, size, page_sizes))
        return;

    ++batch->count;

    batch->total_ranges++;

    batch->biggest_page_size = max(batch->biggest_page_size, biggest_page_size(page_sizes));

    if (tlb_batch_should_invalidate_all(batch))
        return;

    new_entry = &batch->ranges[batch->count - 1];
    new_entry->start = start;
    new_entry->size = size;
    new_entry->page_sizes = page_sizes;
}
