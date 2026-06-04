// Detect system preference and apply dark mode
if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
   document.documentElement.classList.add("dark");
} else {
   document.documentElement.classList.remove("dark");
}

// Get toggle switch
const toggle = document.getElementById("themeToggle");
const htmlEl = document.documentElement;

// Sync toggle UI with current mode
toggle.checked = htmlEl.classList.contains("dark");

// Toggle theme on user action
toggle.addEventListener("change", () => {
   htmlEl.classList.toggle("dark", toggle.checked);
});




// range slider active
document.addEventListener("DOMContentLoaded", () => {
   const sliders = document.querySelectorAll(".range-slider");
   sliders.forEach((slider) => {
      const tooltip = slider.nextElementSibling;
      function updateSlider() {
         const value = +slider.value;
         const max = +slider.max;
         const percent = (value / max) * 100;
         slider.style.background = `linear-gradient(to right, #2A85FF ${percent}%, #272B30 ${percent}%)`;
         tooltip.textContent = value;
         const sliderWidth = slider.offsetWidth;
         const thumbWidth = 16;
         const offset =
            (value / max) * (sliderWidth - thumbWidth) + thumbWidth / 2;
         tooltip.style.left = `${offset}px`;
      }
      slider.addEventListener("input", updateSlider);
      window.addEventListener("resize", updateSlider);
      updateSlider();
   });
});

// modal active
document.addEventListener("DOMContentLoaded", () => {
   // Open modal buttons
   const toggleButtons = document.querySelectorAll(".modal-toggle-btn");

   // Open modal on click
   toggleButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
         const modalId = btn.getAttribute("data-modal-id");
         const modal = document.getElementById(modalId);
         if (modal) modal.classList.add("show"); // SHOW the modal
      });
   });

   // Close modal on .modal-close button
   document.querySelectorAll(".modal-close").forEach((btn) => {
      btn.addEventListener("click", () => {
         const modal = btn.closest(".theme-modal");
         if (modal) modal.classList.remove("show"); // HIDE the modal
      });
   });

   // Close modal on clicking backdrop
   document.querySelectorAll(".theme-modal").forEach((modal) => {
      modal.addEventListener("click", (e) => {
         const dialog = modal.querySelector(".modal-dialog");
         if (!dialog.contains(e.target)) {
            modal.classList.remove("show"); // HIDE the modal
         }
      });
   });
});
