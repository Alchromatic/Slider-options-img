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

      isAuthenticated() {
         return !!this.getToken();
      },

      logout(redirect = 'signin.html') {
         localStorage.removeItem(TOKEN_KEY);
         localStorage.removeItem(USER_KEY);
         if (redirect) window.location.href = redirect;
      },

      /* Redirect to sign-in if there is no token. Call at the top of guarded pages. */
      requireAuth(redirect = 'signin.html') {
         if (!this.isAuthenticated()) {
            window.location.href = redirect;
            return false;
         }
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
         if (!res.ok) throw new Error(data.detail || 'Not authenticated');
         return data.user;
      },
   };

   window.Auth = Auth;
})();
