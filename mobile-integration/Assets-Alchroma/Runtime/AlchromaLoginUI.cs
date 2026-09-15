using System;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace Alchroma
{
    /// <summary>
    /// Self-contained sign-in / pairing overlay built from code (no prefab, no
    /// scene edits). Shown automatically at startup when nobody is signed in
    /// (AlchromaConfig.AutoShowLogin) and again if the device is revoked.
    ///
    ///   phone / tablet : email + password, or "Pair with code" from the Devices page
    ///   headset        : shows the 6-digit code to type on the Devices page
    ///
    /// Call AlchromaLoginUI.Show() / Hide() to drive it yourself, or build your
    /// own UI on top of AlchromaAuth / AlchromaPairing and set AutoShowLogin=false.
    /// </summary>
    public class AlchromaLoginUI : MonoBehaviour
    {
        public static AlchromaLoginUI Instance { get; private set; }

        static readonly Color Bg = new Color(0.07f, 0.075f, 0.08f, 0.96f);
        static readonly Color Panel = new Color(0.11f, 0.12f, 0.13f, 1f);
        static readonly Color Field = new Color(0.17f, 0.18f, 0.2f, 1f);
        static readonly Color Accent = new Color(0.16f, 0.52f, 1f, 1f);
        static readonly Color Muted = new Color(0.62f, 0.65f, 0.68f, 1f);

        Canvas canvas;
        GameObject root, formGroup, codeGroup, headsetGroup;
        InputField emailField, passwordField, codeField;
        Text status, headsetCode;
        Button loginBtn, redeemBtn;
        bool built;

        public static void Show()
        {
            var ui = Instance != null ? Instance : AlchromaApi.Instance.gameObject.AddComponent<AlchromaLoginUI>();
            ui.Open();
        }

        public static void Hide()
        {
            if (Instance != null) Instance.Close();
        }

        void Awake()
        {
            Instance = this;
        }

        void Start()
        {
            AlchromaSession.Load();
            AlchromaSession.Changed += OnSessionChanged;
            AlchromaApi.SessionLost += OnSessionLost;
            AlchromaPairing.CodeReady += OnCodeReady;
            AlchromaPairing.Failed += OnPairFailed;

            if (AlchromaSession.IsLoggedIn)
            {
                AlchromaAuth.Validate(valid => { if (!valid && AlchromaConfig.AutoShowLogin) Open(); });
            }
            else if (AlchromaConfig.AutoShowLogin)
            {
                Open();
            }
        }

        void OnDestroy()
        {
            AlchromaSession.Changed -= OnSessionChanged;
            AlchromaApi.SessionLost -= OnSessionLost;
            AlchromaPairing.CodeReady -= OnCodeReady;
            AlchromaPairing.Failed -= OnPairFailed;
        }

        void OnSessionChanged()
        {
            if (AlchromaSession.IsLoggedIn)
            {
                AlchromaCaptures.FlushQueue();
                Close();
            }
            else if (AlchromaConfig.AutoShowLogin)
            {
                Open();   // signed out (Logout or revoked): offer to sign in again
            }
        }

        void OnSessionLost()
        {
            if (AlchromaConfig.AutoShowLogin) Open();
        }

        void OnCodeReady(string code)
        {
            if (headsetCode != null) headsetCode.text = code.Length == 6 ? code.Substring(0, 3) + " " + code.Substring(3) : code;
            SetStatus("Enter this code on GeoMagic → Devices → Pair a headset", Muted);
        }

        void OnPairFailed(string msg)
        {
            SetStatus(msg, new Color(1f, 0.45f, 0.35f));
        }

        // ------------------------------------------------------------ open/close

        public void Open()
        {
            if (!built) Build();
            root.SetActive(true);
            bool headset = AlchromaDevice.IsHeadset;
            headsetGroup.SetActive(headset);
            formGroup.SetActive(!headset);
            codeGroup.SetActive(false);
            SetStatus(headset ? "Getting a pairing code…" : "", Muted);
            if (headset)
            {
                var pairing = AlchromaPairing.Instance != null ? AlchromaPairing.Instance : AlchromaApi.Instance.gameObject.AddComponent<AlchromaPairing>();
                if (string.IsNullOrEmpty(pairing.CurrentCode)) pairing.StartCodePairing();
                else OnCodeReady(pairing.CurrentCode);
            }
        }

        public void Close()
        {
            if (root != null) root.SetActive(false);
        }

        // ------------------------------------------------------------- actions

        void DoLogin()
        {
            SetStatus("Signing in…", Muted);
            loginBtn.interactable = false;
            AlchromaAuth.Login(emailField.text, passwordField.text, (ok, err) =>
            {
                loginBtn.interactable = true;
                if (!ok) SetStatus(err ?? "Login failed", new Color(1f, 0.45f, 0.35f));
            });
        }

        void DoRedeem()
        {
            SetStatus("Pairing…", Muted);
            redeemBtn.interactable = false;
            var pairing = AlchromaPairing.Instance != null ? AlchromaPairing.Instance : AlchromaApi.Instance.gameObject.AddComponent<AlchromaPairing>();
            pairing.RedeemCode(codeField.text, (ok, err) =>
            {
                redeemBtn.interactable = true;
                if (!ok) SetStatus(err ?? "Pairing failed", new Color(1f, 0.45f, 0.35f));
            });
        }

        void SetStatus(string msg, Color c)
        {
            if (status == null) return;
            status.text = msg ?? "";
            status.color = c;
        }

        // ----------------------------------------------------------------- UI

        void Build()
        {
            built = true;
            bool headset = AlchromaDevice.IsHeadset;
            if (!headset) EnsureEventSystem();   // the headset panel is display-only (the XR rig owns input)

            root = new GameObject("AlchromaLoginUI");
            root.transform.SetParent(transform, false);
            canvas = root.AddComponent<Canvas>();
            root.AddComponent<GraphicRaycaster>();

            if (headset)
            {
                // world-space panel 1.6 m in front of the user (display only)
                canvas.renderMode = RenderMode.WorldSpace;
                var rt = root.GetComponent<RectTransform>();
                rt.sizeDelta = new Vector2(900, 600);
                var cam = Camera.main;
                var origin = cam != null ? cam.transform.position : Vector3.zero;
                var fwd = cam != null ? Vector3.ProjectOnPlane(cam.transform.forward, Vector3.up) : Vector3.forward;
                if (fwd.sqrMagnitude < 0.01f) fwd = Vector3.forward;
                fwd.Normalize();
                root.transform.position = origin + fwd * 1.6f;
                root.transform.rotation = Quaternion.LookRotation(fwd, Vector3.up);
                root.transform.localScale = Vector3.one * 0.0012f;
            }
            else
            {
                canvas.renderMode = RenderMode.ScreenSpaceOverlay;
                canvas.sortingOrder = 30000;
                var scaler = root.AddComponent<CanvasScaler>();
                scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
                scaler.referenceResolution = new Vector2(1080, 1920);
                scaler.matchWidthOrHeight = 0.5f;
            }

            var dim = MakeImage(root.transform, "Dim", Bg);
            Stretch(dim.rectTransform);

            var panel = MakeImage(root.transform, "Panel", Panel);
            var prt = panel.rectTransform;
            prt.anchorMin = new Vector2(0.5f, 0.5f); prt.anchorMax = new Vector2(0.5f, 0.5f);
            prt.sizeDelta = new Vector2(860, 600);
            // height follows the content (the "Pair with code" section can be toggled open)
            var panelFit = panel.gameObject.AddComponent<ContentSizeFitter>();
            panelFit.verticalFit = ContentSizeFitter.FitMode.PreferredSize;

            var layout = panel.gameObject.AddComponent<VerticalLayoutGroup>();
            layout.padding = new RectOffset(48, 48, 48, 48);
            layout.spacing = 18;
            layout.childForceExpandHeight = false;
            layout.childForceExpandWidth = true;
            layout.childControlHeight = true;
            layout.childControlWidth = true;

            MakeText(panel.transform, "Title", "Sign in to GeoMagic", 44, FontStyle.Bold, Color.white, TextAnchor.MiddleCenter, 60);
            MakeText(panel.transform, "Sub", "Use the same account as the web app. Colors you capture will show up in your Paint Collection.", 24, FontStyle.Normal, Muted, TextAnchor.MiddleCenter, 80);

            // ---- phone / tablet: email + password
            formGroup = MakeGroup(panel.transform, "Form");
            emailField = MakeInput(formGroup.transform, "Email", "Email", InputField.ContentType.EmailAddress);
            passwordField = MakeInput(formGroup.transform, "Password", "Password", InputField.ContentType.Password);
            loginBtn = MakeButton(formGroup.transform, "Log in", DoLogin, Accent, 30);
            MakeButton(formGroup.transform, "Pair with a code from the Devices page", () => { codeGroup.SetActive(!codeGroup.activeSelf); }, Field, 24);

            codeGroup = MakeGroup(formGroup.transform, "Code");
            codeField = MakeInput(codeGroup.transform, "PairCode", "Code (e.g. ABCD2345)", InputField.ContentType.Alphanumeric);
            redeemBtn = MakeButton(codeGroup.transform, "Pair this device", DoRedeem, Accent, 28);
            codeGroup.SetActive(false);

            // ---- headset: show the code
            headsetGroup = MakeGroup(panel.transform, "Headset");
            MakeText(headsetGroup.transform, "HsLabel", "Your pairing code", 26, FontStyle.Normal, Muted, TextAnchor.MiddleCenter, 40);
            headsetCode = MakeText(headsetGroup.transform, "HsCode", "··· ···", 96, FontStyle.Bold, Color.white, TextAnchor.MiddleCenter, 120);
            MakeText(headsetGroup.transform, "HsHelp", "On a computer or phone open GeoMagic → Devices → Pair a headset and enter this code.", 24, FontStyle.Normal, Muted, TextAnchor.MiddleCenter, 80);

            status = MakeText(panel.transform, "Status", "", 24, FontStyle.Normal, Muted, TextAnchor.MiddleCenter, 70);
            MakeButton(panel.transform, "Continue without an account", () => Close(), Field, 24);
        }

        // ------------------------------------------------------------ helpers

        static Font uiFont;

        public static Font UiFont
        {
            get
            {
                if (uiFont == null)
                {
                    try { uiFont = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf"); } catch (Exception) { }
                    if (uiFont == null)
                    {
                        try { uiFont = Resources.GetBuiltinResource<Font>("Arial.ttf"); } catch (Exception) { }
                    }
                    if (uiFont == null) uiFont = Font.CreateDynamicFontFromOSFont("Arial", 16);
                }
                return uiFont;
            }
        }

        public static void EnsureEventSystem()
        {
#if UNITY_2023_1_OR_NEWER
            var es = UnityEngine.Object.FindFirstObjectByType<EventSystem>();
#else
            var es = UnityEngine.Object.FindObjectOfType<EventSystem>();
#endif
            if (es != null) return;
            var go = new GameObject("EventSystem");
            go.AddComponent<EventSystem>();
#if ENABLE_INPUT_SYSTEM && !ENABLE_LEGACY_INPUT_MANAGER
            go.AddComponent<UnityEngine.InputSystem.UI.InputSystemUIInputModule>();
#else
            go.AddComponent<StandaloneInputModule>();
#endif
            DontDestroyOnLoad(go);
        }

        static void Stretch(RectTransform rt)
        {
            rt.anchorMin = Vector2.zero; rt.anchorMax = Vector2.one;
            rt.offsetMin = Vector2.zero; rt.offsetMax = Vector2.zero;
        }

        static Image MakeImage(Transform parent, string name, Color color)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var img = go.AddComponent<Image>();
            img.color = color;
            return img;
        }

        static GameObject MakeGroup(Transform parent, string name)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            go.AddComponent<RectTransform>();
            // a VerticalLayoutGroup reports its preferred height to the parent group
            // by itself; no ContentSizeFitter (Unity warns about fitters inside groups)
            var l = go.AddComponent<VerticalLayoutGroup>();
            l.spacing = 14;
            l.childForceExpandHeight = false;
            l.childForceExpandWidth = true;
            l.childControlHeight = true;
            l.childControlWidth = true;
            return go;
        }

        static Text MakeText(Transform parent, string name, string value, int size, FontStyle style, Color color, TextAnchor anchor, float height)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var t = go.AddComponent<Text>();
            t.font = UiFont;
            t.fontSize = size;
            t.fontStyle = style;
            t.color = color;
            t.alignment = anchor;
            t.horizontalOverflow = HorizontalWrapMode.Wrap;
            t.verticalOverflow = VerticalWrapMode.Overflow;
            t.text = value;
            var le = go.AddComponent<LayoutElement>();
            le.preferredHeight = height;
            le.minHeight = height;
            return t;
        }

        static InputField MakeInput(Transform parent, string name, string placeholder, InputField.ContentType type)
        {
            var bg = MakeImage(parent, name, Field);
            var le = bg.gameObject.AddComponent<LayoutElement>();
            le.preferredHeight = 96; le.minHeight = 96;

            var textGo = new GameObject("Text");
            textGo.transform.SetParent(bg.transform, false);
            var text = textGo.AddComponent<Text>();
            text.font = UiFont; text.fontSize = 30; text.color = Color.white; text.alignment = TextAnchor.MiddleLeft;
            text.supportRichText = false;
            var trt = text.rectTransform; Stretch(trt); trt.offsetMin = new Vector2(24, 0); trt.offsetMax = new Vector2(-24, 0);

            var phGo = new GameObject("Placeholder");
            phGo.transform.SetParent(bg.transform, false);
            var ph = phGo.AddComponent<Text>();
            ph.font = UiFont; ph.fontSize = 30; ph.color = Muted; ph.alignment = TextAnchor.MiddleLeft; ph.text = placeholder;
            var prt = ph.rectTransform; Stretch(prt); prt.offsetMin = new Vector2(24, 0); prt.offsetMax = new Vector2(-24, 0);

            var input = bg.gameObject.AddComponent<InputField>();
            input.targetGraphic = bg;
            input.textComponent = text;
            input.placeholder = ph;
            input.contentType = type;
            input.lineType = InputField.LineType.SingleLine;
            return input;
        }

        public static Button MakeButton(Transform parent, string label, Action onClick, Color color, int fontSize)
        {
            var bg = MakeImage(parent, "Button:" + label, color);
            var le = bg.gameObject.AddComponent<LayoutElement>();
            le.preferredHeight = 92; le.minHeight = 92;
            var btn = bg.gameObject.AddComponent<Button>();
            btn.targetGraphic = bg;
            var t = MakeText(bg.transform, "Label", label, fontSize, FontStyle.Bold, Color.white, TextAnchor.MiddleCenter, 92);
            Stretch(t.rectTransform);
            UnityEngine.Object.Destroy(t.GetComponent<LayoutElement>());
            btn.onClick.AddListener(() => onClick?.Invoke());
            return btn;
        }
    }
}
