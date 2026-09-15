using System;
using Newtonsoft.Json;
using UnityEngine;

namespace Alchroma
{
    /// <summary>
    /// Opens the webapp's own pages (Portfolio, Templates, Paint Collection…)
    /// signed in as the current user. The page is opened through
    /// oauth-callback.html, which stores the session in the web view and then
    /// redirects to `next`, so the sidebar pages are exactly the webapp's.
    ///
    /// Default: the system browser (Application.OpenURL). Define ALCHROMA_WEBVIEW
    /// and add the gree/unity-webview package to show the pages inside the app
    /// instead (see AlchromaWebView.cs and the integration guide).
    /// </summary>
    public static class AlchromaWebPages
    {
        public const string Dashboard = "dashboard.html";
        public const string Portfolio = "portfolio.html";
        public const string Templates = "templates.html";
        public const string PaintCollection = "paint-collection.html";
        public const string Devices = "devices.html";
        public const string ColorGuide = "color-guide.html";

        public static string UrlFor(string page)
        {
            page = string.IsNullOrEmpty(page) ? Dashboard : page.TrimStart('/');
            if (!AlchromaSession.IsLoggedIn) return AlchromaConfig.Url("/" + page);
            var user = JsonConvert.SerializeObject(AlchromaSession.User);
            return AlchromaConfig.Url("/oauth-callback.html")
                + "?token=" + Uri.EscapeDataString(AlchromaSession.AccessToken)
                + "&user=" + Uri.EscapeDataString(user)
                + "&next=" + Uri.EscapeDataString(page);
        }

        public static void Open(string page)
        {
            var url = UrlFor(page);
#if ALCHROMA_WEBVIEW
            AlchromaWebView.Show(url);
#else
            Application.OpenURL(url);
#endif
        }

        public static void OpenPortfolio() => Open(Portfolio);
        public static void OpenTemplates() => Open(Templates);
        public static void OpenPaintCollection() => Open(PaintCollection);
        public static void OpenDevices() => Open(Devices);
    }
}
