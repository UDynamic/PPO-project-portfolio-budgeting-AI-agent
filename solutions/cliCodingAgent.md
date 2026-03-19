# cursor + WSL Terminal Not Scrolling When Running GAP/GAPCode
When running GAP/GAPCode inside the VS Code integrated WSL terminal, the terminal becomes stuck at the bottom few lines and cannot scroll up at all (mouse, PgUp, Shift+PgUp all fail).  
Normal shell commands scroll fine — only GAP/GAPCode breaks scrolling.

This issue is caused by GAP switching the terminal into **alternate screen mode**, which VS Code cannot scroll.

---
## Easiest solution:

**use wsl itself instead of IDE terminals**

```  
  - Navigate to your project folder in Explorer.
  - Click the address bar, type wsl, and press Enter.
```

---
## Diagnostics Performed

### 1. Global VS Code settings checked
- `terminal.integrated.scrollback` increased → no effect  
- `terminal.integrated.gpuAcceleration` missing  
- `terminal.integrated.rendererType` missing  
- Shell integration already disabled  
- Terminal type changed from `xterm-256color` to `xterm` → no change  

### 2. Terminal behavior tests
- Normal commands scroll correctly  
- `seq 1 500` scrolls  
- Only GAP/GAPCode output *cannot scroll*  
- GAP CLI always sticks to the bottom  

Conclusion:  
**The problem is not VS Code or WSL — it’s GAP enabling the alternate screen buffer.**

---

## Final Working Solution (Simple, Step‑By‑Step)

### Step 1 — Open a normal WSL terminal (not inside GAP)

You should see:
```
username@ubuntu:~$
```

### Step 2 — Edit `.bashrc`
```
nano ~/.bashrc
```

### Step 3 — Add this line at the bottom
```
export GAP_NO_ALT_SCREENS=1
```

### Step 4 — Save and exit nano
- CTRL+O → Enter  
- CTRL+X  

### Step 5 — Reload shell
```
source ~/.bashrc
```

### Step 6 — Run GAP/GAPCode again
```
gap
```
or
```
gapcode
```

### Step 7 — Test scrolling
```
for i in [1..200] do Print(i,"\n"); od;
```

Scrolling now works normally.
