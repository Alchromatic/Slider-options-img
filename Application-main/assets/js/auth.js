/* Shared auth client — talks to /api/auth/* on this app's own backend.
 * The backend (auth_db.py + auth_routes.py) connects to the same Supabase
 * auth_users table and uses the same JWT_SECRET as the modiqom/multi-model
 * project, so accounts and tokens are interchangeable. */
(function () {
   const API_BASE =
      (new URLSearchParams(location.search).get('api') || '').replace(/\/$/, '') ||
      'https://alchromaticdemo.up.railway.app';

   const TOKEN_KEY = 'gm_access_token';
   const USER_KEY = 'gm_user';

   const Auth = {
      apiBase: API_BASE,

      saveSession(data) {
         if (data && data.access_token) {
            localStorage.setItem(TOKEN_KEY, data.access_token);
         }
         if (data && data.user) {
            localStorage.setItem(USER_KEY, JSON.stringify(data.user));
         }
      },

      getToken() {
         return localStorage.getItem(TOKEN_KEY);
      },

      getUser() {
         try {
            return JSON.parse(localStorage.getItem(USER_KEY) || 'null');
         } catch (e) {
            return null;
         }
      },

      /* True if the JWT's `exp` has passed (tokens last 7 days). Unreadable
         tokens return false and are left for the server to judge. */
      tokenExpired(token) {
         try {
            const part = token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/');
            const payload = JSON.parse(atob(part));
            return typeof payload.exp === 'number' && payload.exp * 1000 <= Date.now();
         } catch (e) {
            return false;
         }
      },

      /* Signed in = a saved token that hasn't expired. An expired one is dropped,
         so the sign-in page shows its form instead of bouncing to the app. */
      isAuthenticated() {
         const token = this.getToken();
         if (token && this.tokenExpired(token)) {
            this.clearSession();
            this.sessionExpired = true;
            return false;
         }
         return !!token;
      },

      clearSession() {
         localStorage.removeItem(TOKEN_KEY);
         localStorage.removeItem(USER_KEY);
      },

      logout(redirect = 'signin.html') {
         this.clearSession();
         // the open composition (and other per-tab state) belongs to this user
         try { sessionStorage.clear(); } catch (e) {}
         if (redirect) window.location.href = redirect;
      },

      /* Redirect to sign-in if there is no token. Call at the top of guarded pages. */
      requireAuth(redirect = 'signin.html') {
         if (!this.isAuthenticated()) {
            window.location.href = this.sessionExpired
               ? redirect + (redirect.includes('?') ? '&' : '?') + 'expired=1'
               : redirect;
            return false;
         }
         return true;
      },

      /* For the sign-in / sign-up pages: go to `target` only if the server still
         accepts the saved session. A rejected one (expired, or signed with an old
         secret) is cleared so the form can be used. Resolves true if redirecting. */
      async redirectIfSignedIn(target) {
         if (!this.isAuthenticated()) return false;
         try {
            await this.me();
         } catch (err) {
            if (err.status === 401 || err.status === 404) {
               this.clearSession();
               this.sessionExpired = true;
               return false;
            }
            // offline or server error: keep going, as before
         }
         window.location.href = target;
         return true;
      },

      async register({ email, password, name, organization_name }) {
         const res = await fetch(`${API_BASE}/api/auth/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password, name, organization_name }),
         });
         const data = await res.json().catch(() => ({}));
         if (!res.ok) throw new Error(data.detail || 'Registration failed');
         this.saveSession(data);
         return data;
      },

      async login({ email, password }) {
         const res = await fetch(`${API_BASE}/api/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password }),
         });
         const data = await res.json().catch(() => ({}));
         if (!res.ok) throw new Error(data.detail || 'Invalid email or password');
         this.saveSession(data);
         return data;
      },

      async me() {
         const token = this.getToken();
         if (!token) throw new Error('Not authenticated');
         const res = await fetch(`${API_BASE}/api/auth/me`, {
            headers: { Authorization: `Bearer ${token}` },
         });
         const data = await res.json().catch(() => ({}));
         if (!res.ok) {
            const err = new Error(data.detail || 'Not authenticated');
            err.status = res.status;
            throw err;
         }
         return data.user;
      },
   };

   window.Auth = Auth;
})();
