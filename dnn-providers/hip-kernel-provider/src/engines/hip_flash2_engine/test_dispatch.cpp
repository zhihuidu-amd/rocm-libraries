// Verify the C++ dispatch rule reproduces the Python rule that produced 1.043x.
#include "Flash2Dispatch.hpp"
#include <cstdio>
#include <cstring>
using namespace hip_flash2_engine;

struct Case
{
    int B, H, S, D;
    bool c;
    const char* want;
    int wantSplit;
};

int main()
{
    // Expected values taken from the measured 21-shape run (combined.py "pick" column).
    Case cases[] = {
        {1, 32, 512, 128, true, "w4q1k4", 1},
        {1, 32, 1024, 128, true, "w8q1k4", 1},
        {1, 32, 2048, 128, true, "w8q2k4", 1},
        {1, 32, 4096, 128, true, "w8q3k2", 1},
        {1, 32, 2048, 128, false, "w8q2k4", 1},
        {1, 32, 4096, 128, false, "w8q2k4", 1},
        {1, 32, 2048, 64, true, "w8q2k4", 1},
        {4, 32, 1024, 128, true, "w8q3k2", 1},
        {1, 32, 8192, 128, true, "w8q3k2", 1},
        {1, 32, 8192, 128, false, "w8q3k2", 1},
        {2, 16, 3072, 128, false, "w8q3k2", 1},
        {1, 64, 1536, 64, true, "w8q2k4", 1},
        {8, 32, 512, 128, true, "w8q2k4", 1},
        {1, 16, 6144, 128, true, "w8q3k2", 1},
        {1, 32, 3072, 128, true, "w8q3k2", 1},
        {2, 8, 8192, 128, true, "w8q3k2", 1},
        {16, 32, 256, 128, true, "w8q2k4", 1},
        {1, 8, 2048, 128, false, "w8q2k4", 4}, // split-K path
        {2, 32, 2048, 128, true, "w8q3k2", 1},
        {1, 40, 1024, 128, false, "w8q2k4", 1},
        {4, 16, 4096, 64, true, "w8q3k4", 1},
    };
    int fail = 0;
    printf("%-26s %-10s %-10s %6s %6s  %s\n", "shape", "want", "got", "wantSK", "gotSK", "");
    for(const auto& t : cases)
    {
        auto s = selectFlash2Config(t.B, t.H, t.S, t.D, t.c);
        const bool ok = (std::strcmp(s.variant.tag, t.want) == 0) && (s.splitK == t.wantSplit);
        if(!ok)
            fail++;
        char lbl[64];
        snprintf(lbl, sizeof lbl, "B%dH%dS%dD%d%c", t.B, t.H, t.S, t.D, t.c ? 'c' : 'n');
        printf("%-26s %-10s %-10s %6d %6d  %s\n",
               lbl,
               t.want,
               s.variant.tag,
               t.wantSplit,
               s.splitK,
               ok ? "ok" : "MISMATCH");
        // geometry sanity: qPerCta must be waves*QG*16 and blockDim waves*64
        if(s.variant.blockDim % 64 != 0)
        {
            printf("   bad blockDim\n");
            fail++;
        }
    }
    // workspace sanity
    size_t ws = flash2WorkspaceBytes(1, 8, 2048, 128, 4);
    size_t expect = (size_t)1 * 8 * 4 * 2048 * 128 * 4 + 2 * (size_t)1 * 8 * 4 * 2048 * 4;
    printf("\nworkspace(B1 H8 S2048 D128 split4) = %zu bytes (expect %zu) %s\n",
           ws,
           expect,
           ws == expect ? "ok" : "MISMATCH");
    if(ws != expect)
        fail++;
    printf("\n%s: %d mismatch(es)\n", fail ? "FAIL" : "PASS", fail);
    return fail ? 1 : 0;
}
