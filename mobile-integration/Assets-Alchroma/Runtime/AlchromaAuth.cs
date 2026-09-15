using System;
using Newtonsoft.Json.Linq;

namespace Alchroma
{
    /// <summary>
    /// Account actions. Login uses the same /api/auth/login as the webapp, then
    /// registers this device to swap the 7-day user token for a 90-day device
    /// token (revocable from the webapp Devices page).
    /// </summary>
    public static class AlchromaAuth
    {
        /// <summary>done(success, errorMessage)</summary>
        public static void Login(string email, string password, Action<bool, string> done)
        {
            email = (email ?? "").Trim();
            if (string.IsNullOrEmpty(email) || string.IsNullOrEmpty(password))
            {
                done?.Invoke(false, "Enter your email and password");
                return;
            }
            AlchromaApi.Post("/api/auth/login", new { email, password }, res =>
            {
                if (!res.ok || res.json == null)
                {
                    done?.Invoke(false, res.Detail ?? "Login failed");
                    return;
                }
                var user = res.json["user"]?.ToObject<AlchromaUser>();
                AlchromaSession.Set((string)res.json["access_token"], "user", user);
                // Registration failing (offline, etc.) must not undo a successful login.
                RegisterDevice((ok, err) => done?.Invoke(true, null));
            }, auth: false);
        }

        /// <summary>Register / refresh this device for the signed-in user and store the device token.</summary>
        public static void RegisterDevice(Action<bool, string> done)
        {
            if (!AlchromaSession.IsLoggedIn)
            {
                done?.Invoke(false, "not_logged_in");
                return;
            }
            AlchromaApi.Post("/api/devices/register", AlchromaDevice.Info(), res =>
            {
                if (!res.ok || res.json == null)
                {
                    done?.Invoke(false, res.Detail ?? "Device registration failed");
                    return;
                }
                ApplyDeviceSession(res.json);
                done?.Invoke(true, null);
            });
        }

        /// <summary>Store a {access_token, user, device} payload from /register, /pair/redeem or /pair/status.</summary>
        public static void ApplyDeviceSession(JObject payload)
        {
            if (payload == null) return;
            var user = payload["user"] != null && payload["user"].Type == JTokenType.Object
                ? payload["user"].ToObject<AlchromaUser>()
                : AlchromaSession.User;
            AlchromaSession.Set((string)payload["access_token"], (string)payload["token_type"] ?? "device", user);
        }

        /// <summary>Check the stored token with the backend. done(valid)</summary>
        public static void Validate(Action<bool> done)
        {
            if (!AlchromaSession.IsLoggedIn)
            {
                done?.Invoke(false);
                return;
            }
            if (AlchromaSession.TokenType == "device")
            {
                // heartbeat also fails with device_revoked when the webapp unpaired us
                AlchromaApi.Post("/api/devices/heartbeat", null, res => done?.Invoke(res.ok || res.status != 401));
                return;
            }
            AlchromaApi.Get("/api/auth/me", res =>
            {
                if (res.ok && res.json?["user"] != null)
                {
                    var fresh = res.json["user"].ToObject<AlchromaUser>();
                    if (fresh != null && fresh.id != null && (AlchromaSession.User == null || fresh.email != AlchromaSession.User.email))
                        AlchromaSession.Set(AlchromaSession.AccessToken, AlchromaSession.TokenType, fresh);
                    done?.Invoke(true);
                }
                else
                {
                    // network errors keep the cached session; a definite 401 has already cleared it
                    done?.Invoke(res.status != 401);
                }
            });
        }

        public static void Logout()
        {
            AlchromaSession.Clear();
        }
    }
}
