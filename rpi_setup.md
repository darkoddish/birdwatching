# Raspberry Pi Setup for Birdwatch AI

### 🌐 Initial Configuration

1. Select Country: **United States**
2. Select Language: **American English**
3. Select Timezone: **Eastern**
4. Create Username and Password
5. Connect to Wi-Fi
6. Choose Browser: **Chromium**

### 🔄 System Update

7. Open Terminal
8. Run:
   ```bash
   sudo apt update && sudo apt full-upgrade -y
   ```
9. If prompted about config file conflicts, select: **Keep package maintainer's version**
10. Reboot:
    ```bash
    sudo reboot
    ```
11. If prompted to choose window manager, select: **Wayfire**

### 🖥️ Record System Info

12. Run the following commands and record outputs:

- OS version:

  ```bash
  cat /etc/os-release
  ```

  → Debian GNU/Linux 12 (bookworm)

- Kernel version:

  ```bash
  uname -r
  ```

  → 6.12.34+rpt-rpi-2712

- Python version:

  ```bash
  python3 --version
  ```

  → Python 3.11.2

- Hostname & IP:

  ```bash
  hostname -I
  ```

  → 192.168.1.134

- CPU info:

  ```bash
  lscpu
  ```

  → Cortex-A76, 4 cores, 64-bit, max 2.4GHz

- Memory info:

  ```bash
  free -h
  ```

  → 7.9Gi total, 6.6Gi free

- Disk usage:

  ```bash
  df -h
  ```

  → 117G total on root (/), 5% used

### 📁 Clone GitHub Project

13. Install Git:

```bash
sudo apt install git -y
```

14. Navigate to desired directory:

```bash
cd ~/Documents/projects
```

15. Clone your repository:

```bash
git clone https://github.com/darkoddish/birdwatching
```

16. Enter project directory:

```bash
cd birdwatching
```

### 🐍 Create Python Virtual Environment

17. Install venv (if not installed):

```bash
sudo apt install python3-venv -y
```

18. Create virtual environment:

```bash
python3 -m venv venv_bw
```

19. Activate virtual environment:

```bash
source venv_bw/bin/activate
```

20. Upgrade pip:

```bash
pip install --upgrade pip
```

