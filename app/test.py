from onvif import ONVIFCamera

camera = ONVIFCamera('192.168.1.6', 8000, 'admin', 'Kang@r00')
media_service = camera.create_media_service()
ptz_service = camera.create_ptz_service()

profiles = media_service.GetProfiles()
print("Found profiles:", [p.Name for p in profiles])

ptz_config = profiles[0].PTZConfiguration
print("PTZ config:", ptz_config)