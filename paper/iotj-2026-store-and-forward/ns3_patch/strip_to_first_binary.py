import re,sys
s=open(sys.argv[1]).read()
def rm(a):
    global s
    assert a in s,a[:50]
    s=s.replace(a,'',1)
rm('bool g_upRoundRobin = true;                  // true: one report per device per pass; false: exhaustive backlog\n')
rm("std::vector<uint32_t> g_rrNext;              // [sat]: device index the satellite's next pass starts from\n")
rm('int64_t g_rrGrant = -1;                      // device holding the one-report grant of round robin, -1 = none\n')
rm('''        if (g_upRoundRobin)
        {
            if (g_rrGrant != static_cast<int64_t>(d))
            {
                return false; // round robin: a report leaves only under the scheduler's grant
            }
            g_rrGrant = -1; // one report, so the drain stops at the next refusal
        }
''')
i=s.index('/// Per-report round robin of the shared uplink'); j=s.index('void\nTick()')
s=s[:i]+s[j:]
s=s.replace('''        if (g_upBps <= 0.0 || !g_upRoundRobin)
        {
            g_devQ[d]->SetContactUp(any && ChooseSatellite(d) >= 0);
        }
''','''        g_devQ[d]->SetContactUp(any && ChooseSatellite(d) >= 0);
''',1)
rm('''    if (g_upBps > 0.0 && g_upRoundRobin)
    {
        ServeUplinkRoundRobin(); // after the loop, so every device's elevations are those of this tick
    }
''')
s=re.sub(r'    cmd\.AddValue\("upRoundRobin".*\n','',s,1)
rm('    g_rrNext.assign(S, 0);\n')
open(sys.argv[2],'w').write(s)
