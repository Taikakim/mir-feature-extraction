# MIR Pipeline Feature Benchmark

Generated: 2026-03-30 20:41
Hardware:  AMD Ryzen 9 9900X 12-Core Processor  ·  GPU[0]		: Card Series: 		AMD Radeon RX 9070 XT
CPU workers: 10

- **PASS 1 – CPU Features**: 200 crops, avg 11.4 s audio
- **PASS 2 – AudioBox Aesthetics**: 20 crops, avg 11.4 s audio
- **PASS 3 – Essentia Classification**: 20 crops, avg 11.4 s audio
- **PASS 4 – Music Flamingo**: 11 crops, avg 11.4 s audio

---
| Feature | N | Mean (s/s) | Std (s/s) | p10–p90 (s/s) | RTF | % total cost |
|---------|---|------------|-----------|---------------|-----|--------------|
| ***PASS 1 – CPU Features*** | | | | | | |
| io.read_crop | 200 | 0.0010 | 0.0004 | 0.0006–0.0014 | 1.0kx | 0.1% |
| loudness | 200 | 0.0055 | 0.0004 | 0.0051–0.0061 | 182x | 0.3% |
| spectral | 200 | 0.0019 | 0.0003 | 0.0017–0.0023 | 518x | 0.1% |
| multiband_rms | 200 | 0.0013 | 0.0002 | 0.0011–0.0015 | 770x | 0.1% |
| saturation | 200 | 0.0000 | 0.0000 | 0.0000–0.0000 | 41.8kx | 0.0% |
| hpcp_tiv | 200 | 0.0010 | 0.0001 | 0.0009–0.0012 | 1.0kx | 0.1% |
| timbral.hardness | 200 | 0.1778 | 0.0163 | 0.1559–0.1966 | 6x | 10.1% |
| timbral.sharpness | 200 | 0.0345 | 0.0114 | 0.0200–0.0481 | 29x | 2.0% |
| timbral.booming | 200 | 0.0518 | 0.0088 | 0.0398–0.0626 | 19x | 2.9% |
| timbral.warmth | 200 | 0.0639 | 0.0142 | 0.0454–0.0848 | 16x | 3.6% |
| timbral.roughness | 200 | 0.0263 | 0.0115 | 0.0147–0.0459 | 38x | 1.5% |
| timbral.depth | 200 | 0.0353 | 0.0121 | 0.0269–0.0390 | 28x | 2.0% |
| timbral.brightness | 200 | 0.0159 | 0.0040 | 0.0120–0.0202 | 63x | 0.9% |
| timbral_total | 200 | 0.4054 | 0.0341 | 0.3722–0.4328 | 2x | 23.0% |
| ***PASS 2 – AudioBox Aesthetics*** | | | | | | |
| audiobox | 20 | 0.0942 | 0.1471 | 0.0207–0.3885 | 11x | 5.3% |
| ***PASS 3 – Essentia Classification*** | | | | | | |
| essentia.audio_load | 20 | 0.0038 | 0.0002 | 0.0036–0.0041 | 264x | 0.2% |
| essentia.danceability | 20 | 0.0479 | 0.2040 | 0.0010–0.0017 | 21x | 2.7% |
| essentia.atonality | 20 | 0.0399 | 0.1694 | 0.0010–0.0010 | 25x | 2.3% |
| essentia.voice | 20 | 0.0395 | 0.1675 | 0.0010–0.0012 | 25x | 2.2% |
| essentia.gmi | 20 | 0.1230 | 0.5319 | 0.0009–0.0010 | 8x | 7.0% |
| ***PASS 4 – Music Flamingo*** | | | | | | |
| flamingo.full | 11 | 0.5962 | 0.2166 | 0.2579–0.8214 | 2x | 33.8% |
| **Total serial / audio-s** | | **1.7661** | | | | 100% |
| *Est. CPU throughput (10 workers)* | | | | | *~12x* | |
