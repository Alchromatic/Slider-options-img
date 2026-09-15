using System;
using System.Collections;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace Alchroma
{
    /// <summary>
    /// Pairs this device to a GeoMagic account without typing a password:
    ///
    ///   * Code pairing (headset / projector): StartCodePairing() asks the backend
    ///     for a 6-digit code, raises CodeReady so you can show it, and polls until
    ///     the user enters the code on the webapp Devices page.
    ///   * Redeem (phone / tablet): the webapp shows a code + QR; RedeemCode() or a
    ///     deep link "alchroma://pair?code=XXXXXXXX" finishes the pairing.
    ///
    /// Also sends the heartbeat that keeps the device "online" on the Devices page.
    /// Added automatically to the [Alchroma] object by AlchromaBootstrap.
    /// </summary>
    public class AlchromaPairing : MonoBehaviour
    {
        public static AlchromaPairing Instance { get; private set; }

        /// <summary>Show this code to the user (headset flow).</summary>
        public static event Action<string> CodeReady;
        /// <summary>Pairing finished; AlchromaSession is now signed in.</summary>
        public static event Action Paired;
        /// <summary>Something went wrong (message is user-readable).</summary>
        public static event Action<string> Failed;

        public string CurrentCode { get; private set; }

        string pollSecret;
        float pollInterval = 3f;
        int expiresIn = 600;
        Coroutine polling;
        Coroutine heartbeat;
        string lastDeepLink;
        int startSeq;   // ignores the reply of a /pair/start that was superseded

        void Awake()
        {
            Instance = this;
        }

        void OnEnable()
        {
            Application.deepLinkActivated += OnDeepLink;
            if (!string.IsNullOrEmpty(Application.absoluteURL)) OnDeepLink(Application.absoluteURL);
            heartbeat = StartCoroutine(Heartbeat());
        }

        void OnDisable()
        {
            Application.deepLinkActivated -= OnDeepLink;
            if (heartbeat != null) StopCoroutine(heartbeat);
            StopCodePairing();
        }

        // ------------------------------------------------------------ code flow

        public void StartCodePairing()
        {
            StopCodePairing();
            int seq = ++startSeq;
            AlchromaApi.Post("/api/devices/pair/start", AlchromaDevice.Info(), res =>
            {
                if (seq != startSeq || AlchromaSession.IsLoggedIn) return;
                if (!res.ok || res.json == null)
                {
                    Failed?.Invoke(res.Detail ?? "Could not get a pairing code");
                    return;
                }
                CurrentCode = (string)res.json["code"];
                pollSecret = (string)res.json["poll_secret"];
                pollInterval = Mathf.Max(2f, (float?)res.json["interval"] ?? 3f);
                expiresIn = (int?)res.json["expires_in"] ?? 600;
                CodeReady?.Invoke(CurrentCode);
                polling = StartCoroutine(Poll(CurrentCode, pollSecret));
            }, auth: false);
        }

        public void StopCodePairing()
        {
            startSeq++;
            if (polling != null)
            {
                StopCoroutine(polling);
                polling = null;
            }
        }

        IEnumerator Poll(string code, string secret)
        {
            float deadline = Time.realtimeSinceStartup + expiresIn;
            while (Time.realtimeSinceStartup < deadline)
            {
                yield return new WaitForSecondsRealtime(pollInterval);

                bool waiting = true;
                string status = "error";
                JObject payload = null;
                AlchromaApi.Get("/api/devices/pair/status?code=" + Uri.EscapeDataString(code) + "&poll_secret=" + Uri.EscapeDataString(secret), res =>
                {
                    if (res.ok && res.json != null) status = (string)res.json["status"] ?? "error";
                    else if (res.status == 404) status = "expired";
                    payload = res.json;
                    waiting = false;
                }, auth: false);
                while (waiting) yield return null;

                if (status == "claimed")
                {
                    polling = null;
                    AlchromaAuth.ApplyDeviceSession(payload);
                    CurrentCode = null;
                    Paired?.Invoke();
                    yield break;
                }
                if (status == "expired" || status == "consumed")
                {
                    break;
                }
                // "pending" or a transient error: keep polling
            }
            polling = null;
            Failed?.Invoke("The pairing code expired — here is a new one");
            StartCodePairing();
        }

        // ---------------------------------------------------------- redeem flow

        /// <summary>Redeem a code from the webapp Devices page (typed, scanned or deep-linked). done(success, error)</summary>
        public void RedeemCode(string code, Action<bool, string> done = null)
        {
            code = (code ?? "").Trim();
            if (code.Length < 4)
            {
                done?.Invoke(false, "Enter the code shown on the Devices page");
                return;
            }
            var body = AlchromaDevice.Info();
            body["code"] = code;
            AlchromaApi.Post("/api/devices/pair/redeem", body, res =>
            {
                if (!res.ok || res.json == null)
                {
                    var msg = res.status == 404 ? "Code not found or expired — get a new one from the Devices page" : (res.Detail ?? "Pairing failed");
                    Failed?.Invoke(msg);
                    done?.Invoke(false, msg);
                    return;
                }
                StopCodePairing();
                AlchromaAuth.ApplyDeviceSession(res.json);
                Paired?.Invoke();
                done?.Invoke(true, null);
            }, auth: false);
        }

        void OnDeepLink(string url)
        {
            if (string.IsNullOrEmpty(url) || url == lastDeepLink) return;
            lastDeepLink = url;
            var code = CodeFromUrl(url);
            if (string.IsNullOrEmpty(code)) return;
            if (AlchromaConfig.VerboseLogs) Debug.Log("[Alchroma] Deep link pairing code received");
            RedeemCode(code);
        }

        /// <summary>Extracts ?code= from "alchroma://pair?code=X" or "https://host/pair.html?code=X".</summary>
        public static string CodeFromUrl(string url)
        {
            try
            {
                var q = url.IndexOf('?');
                if (q < 0) return null;
                foreach (var kv in url.Substring(q + 1).Split('&'))
                {
                    var eq = kv.IndexOf('=');
                    if (eq <= 0) continue;
                    if (kv.Substring(0, eq) == "code")
                        return Uri.UnescapeDataString(kv.Substring(eq + 1).Split('#')[0]);
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning("[Alchroma] Could not parse deep link: " + e.Message);
            }
            return null;
        }

        // ------------------------------------------------------------ heartbeat

        IEnumerator Heartbeat()
        {
            while (true)
            {
                yield return new WaitForSecondsRealtime(Mathf.Max(15f, AlchromaConfig.HeartbeatSeconds));
                if (AlchromaSession.IsLoggedIn && AlchromaSession.TokenType == "device")
                    AlchromaApi.Post("/api/devices/heartbeat", null, null);
            }
        }
    }
}
