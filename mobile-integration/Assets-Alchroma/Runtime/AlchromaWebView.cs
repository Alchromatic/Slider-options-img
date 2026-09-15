// In-app web view for the webapp pages. Optional: only compiled when the
// ALCHROMA_WEBVIEW scripting define is set AND the gree/unity-webview package is
// in the project (Packages/manifest.json:
//   "net.gree.unity-webview": "https://github.com/gree/unity-webview.git?path=/dist/package-nofragment"
// ). Without the define AlchromaWebPages.Open falls back to the system browser.
#if ALCHROMA_WEBVIEW
using UnityEngine;
using UnityEngine.UI;

namespace Alchroma
{
    public class AlchromaWebView : MonoBehaviour
    {
        static AlchromaWebView instance;
        WebViewObject web;
        Canvas bar;

        const int BarHeight = 96;

        public static void Show(string url)
        {
            if (instance == null)
            {
                var go = new GameObject("[AlchromaWebView]");
                DontDestroyOnLoad(go);
                instance = go.AddComponent<AlchromaWebView>();
                instance.Build();
            }
            instance.Load(url);
        }

        public static void Hide()
        {
            if (instance == null) return;
            instance.web.SetVisibility(false);
            instance.bar.gameObject.SetActive(false);
        }

        void Build()
        {
            web = new GameObject("WebViewObject").AddComponent<WebViewObject>();
            web.Init(
                cb: msg => { },
                err: msg => Debug.LogWarning("[Alchroma] WebView error: " + msg),
                httpErr: msg => Debug.LogWarning("[Alchroma] WebView HTTP error: " + msg),
                ld: msg => web.SetVisibility(true),
                enableWKWebView: true);
            web.SetMargins(0, BarHeight, 0, 0);

            // simple top bar with a close button so the user can get back to the app
            var barGo = new GameObject("AlchromaWebViewBar");
            barGo.transform.SetParent(transform, false);
            bar = barGo.AddComponent<Canvas>();
            bar.renderMode = RenderMode.ScreenSpaceOverlay;
            bar.sortingOrder = 32000;
            barGo.AddComponent<GraphicRaycaster>();
            AlchromaLoginUI.EnsureEventSystem();

            var bg = new GameObject("Bar").AddComponent<Image>();
            bg.transform.SetParent(barGo.transform, false);
            bg.color = new Color(0.07f, 0.075f, 0.08f, 1f);
            var rt = bg.rectTransform;
            rt.anchorMin = new Vector2(0, 1); rt.anchorMax = new Vector2(1, 1); rt.pivot = new Vector2(0.5f, 1);
            rt.sizeDelta = new Vector2(0, BarHeight);

            var close = AlchromaLoginUI.MakeButton(bg.transform, "Close", () => Hide(), new Color(0.16f, 0.52f, 1f), 28);
            var crt = close.GetComponent<RectTransform>();
            crt.anchorMin = new Vector2(1, 0.5f); crt.anchorMax = new Vector2(1, 0.5f); crt.pivot = new Vector2(1, 0.5f);
            crt.anchoredPosition = new Vector2(-16, 0); crt.sizeDelta = new Vector2(180, 64);
        }

        void Load(string url)
        {
            bar.gameObject.SetActive(true);
            web.LoadURL(url);
            web.SetVisibility(true);
        }
    }
}
#endif
