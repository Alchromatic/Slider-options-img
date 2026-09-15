using UnityEngine;

namespace Alchroma
{
    /// <summary>
    /// Zero-setup entry point: after the first scene loads this creates the
    /// persistent [Alchroma] object with the API client, the pairing / heartbeat
    /// component and (unless AlchromaConfig.AutoShowLogin is false) the sign-in
    /// overlay, then flushes any captures that were queued while signed out.
    /// Nothing needs to be added to a scene.
    /// </summary>
    public static class AlchromaBootstrap
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Init()
        {
            var host = AlchromaApi.Instance.gameObject;
            if (host.GetComponent<AlchromaPairing>() == null) host.AddComponent<AlchromaPairing>();
            if (host.GetComponent<AlchromaLoginUI>() == null) host.AddComponent<AlchromaLoginUI>();
            AlchromaCaptures.FlushQueue();
            if (AlchromaConfig.VerboseLogs)
                Debug.Log($"[Alchroma] ready — backend {AlchromaConfig.BaseUrl}, device {AlchromaDevice.Type}/{AlchromaDevice.Platform}, signed in: {AlchromaSession.IsLoggedIn}");
        }
    }
}
