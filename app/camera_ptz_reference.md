# ONVIF Camera Capability Summary

## ✅ Home Position Support

- **Supported:** Yes (`GotoHomePosition` and `SetHomePosition` are available)

---

## 🎥 Media Profiles

### Profile: `Profile000_MainStream`

- **Token:** `000`
- **Resolution:** 3840x2160 (4K)
- **Encoding:** H264, Profile: Main
- **Frame Rate:** 25 fps
- **Bitrate:** 6144 kbps
- **PTZ Configuration:** PtzConfig000
  - Velocity range: `-1.0 to 1.0` for pan/tilt/zoom
  - Default PTZ speed: PanTilt (1.0, 1.0), Zoom: 1.0
  - Timeout: 5 seconds

### Profile: `Profile001_SubStream`

- **Token:** `001`
- **Resolution:** 640x360
- **Encoding:** H264, Profile: Main
- **Frame Rate:** 10 fps
- **Bitrate:** 256 kbps
- **PTZ Configuration:** Same as MainStream (PtzConfig000)

---

## 🔧 PTZ Capabilities

```json
{
  "Reverse": true,
  "GetCompatibleConfigurations": true,
  "MoveStatus": true
}
```

---

## 📐 PTZ Configuration Options

- **Continuous Pan/Tilt Velocity Space:** `[-1.0, 1.0]`
- **PanTilt Speed Space:** `[0.0, 1.0]`
- **PTZ Timeout Range:** 1s to 10s
- **Zoom options:** Not supported in this config

---

## 📺 Video Source Configuration

- **Bounds:** x: 0, y: 0, width: 3840, height: 2160

---

## 🎚️ Audio

- **Encoding:** AAC
- **Bitrate:** 16 kbps
- **Sample Rate:** 16 kHz

---

## 📡 Multicast Settings

- **IP:** 239.0.1.0 (Video)
- **Port:** 4000
- **TTL:** 64
- **AutoStart:** False

---

## 📈 Video Analytics

- **Motion Module:** `MyCellMotionModule`
  - Sensitivity: 60
- **Rules:** `MyMotionDetectorRule`
  - MinCount: 20
  - Alarm Delays: On 1008 ms, Off 993 ms
  - ActiveCells: base64 encoded (0P8A8A==)

---

## 🧠 PTZ Methods Available

```
clone, create_type, daemon, encrypt, passwd, service_wrapper, to_dict, url, user, ws_client, xaddr, zeep_client
```

---

## 📚 Notes

- **Both profiles share the same PTZ config (token: **``**), and are likely locked to a single node.**
- **Zoom velocity control is listed but not available in PTZ options — may be firmware-dependent or disabled.**

