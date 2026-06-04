// tab start
document.addEventListener("DOMContentLoaded", function () {
  const tabAreas = document.querySelectorAll(".tab-area");

  tabAreas.forEach((area) => {
    const tabButtons = area.querySelectorAll(".tab-btn");
    const tabContents = area.querySelectorAll(".tab-content");

    function activateTab(button) {
      const targetId = button.getAttribute("data-tab");

      // Deactivate only buttons/contents inside this area (avoid nested conflicts)
      tabButtons.forEach((btn) => {
        if (btn.closest(".tab-area") === area) {
          btn.classList.remove("active");
        }
      });

      tabContents.forEach((content) => {
        if (content.closest(".tab-area") === area) {
          content.classList.remove("active");
        }
      });

      // Activate the selected tab and its content
      button.classList.add("active");
      const contentToShow = area.querySelector(`#${targetId}`);
      if (contentToShow && contentToShow.closest(".tab-area") === area) {
        contentToShow.classList.add("active");
      }
    }

    // Set up click listeners
    tabButtons.forEach((btn) => {
      if (btn.closest(".tab-area") === area) {
        btn.addEventListener("click", () => activateTab(btn));
      }
    });

    // Initialize default tab
    const preActiveBtn = Array.from(tabButtons).find((btn) => btn.classList.contains("active") && btn.closest(".tab-area") === area);
    if (preActiveBtn) {
      activateTab(preActiveBtn);
    } else if (tabButtons.length) {
      const firstBtn = Array.from(tabButtons).find((btn) => btn.closest(".tab-area") === area);
      if (firstBtn) activateTab(firstBtn);
    }
  });
});
// tab end

// collapse start
document.addEventListener("DOMContentLoaded", () => {
  const toggles = document.querySelectorAll(".collapse-toggle");
  const boxes = document.querySelectorAll(".collapse-box");

  // -----------------------------
  // INIT HEIGHTS (JS controls all)
  // -----------------------------
  boxes.forEach((box) => {
    if (box.classList.contains("show")) {
      // open by default
      box.style.height = box.scrollHeight + "px";
      box.addEventListener(
        "transitionend",
        function initOpen(e) {
          if (e.propertyName !== "height") return;
          box.style.height = "auto";
          box.removeEventListener("transitionend", initOpen);
        },
        { once: false },
      );
    } else {
      box.style.height = "0px";
    }
  });

  // -----------------------------
  // CLICK TO TOGGLE
  // -----------------------------
  toggles.forEach((btn) => {
    btn.addEventListener("click", () => {
      const selector = btn.getAttribute("data-collapse-target");
      if (!selector) return;

      const box = document.querySelector(selector);
      const isOpen = box.classList.contains("show");

      if (!isOpen) {
        // OPEN
        box.classList.add("show");
        box.style.height = box.scrollHeight + "px";

        box.addEventListener("transitionend", function endOpen(e) {
          if (e.propertyName !== "height") return;
          if (box.classList.contains("show")) {
            box.style.height = "auto";
          }
          box.removeEventListener("transitionend", endOpen);
        });

        btn.classList.add("active"); // FIXED: always set active immediately
      } else {
        // CLOSE
        box.style.height = box.scrollHeight + "px";
        void box.offsetHeight;
        box.style.height = "0px";

        box.addEventListener("transitionend", function endClose(e) {
          if (e.propertyName !== "height") return;
          box.classList.remove("show");
          box.removeEventListener("transitionend", endClose);
        });

        btn.classList.remove("active"); // FIXED: always remove active immediately
      }
    });
  });
});
// collapse end

// dropdown start
document.addEventListener("DOMContentLoaded", function () {
  const toggles = document.querySelectorAll(".dropdown-toggle");

  toggles.forEach((toggle) => {
    const dropdown = toggle.nextElementSibling;

    // Toggle show class on button click
    toggle.addEventListener("click", function (e) {
      e.stopPropagation();

      // Close all other dropdowns
      document.querySelectorAll(".dropdown-menu.show").forEach((open) => {
        if (open !== dropdown) open.classList.remove("show");
      });

      dropdown.classList.toggle("show");
    });

    // Prevent clicks inside the menu from closing it
    dropdown.addEventListener("click", (e) => e.stopPropagation());
  });

  // Global click to close any open dropdown
  document.addEventListener("click", () => {
    document.querySelectorAll(".dropdown-menu.show").forEach((menu) => {
      menu.classList.remove("show");
    });
  });
});
// dropdown end

// range slider start
document.addEventListener("DOMContentLoaded", () => {
  const sliders = document.querySelectorAll(".range-slider");
  sliders.forEach((slider) => {
    const tooltip = slider.nextElementSibling;
    function updateSlider() {
      const value = +slider.value;
      const max = +slider.max;
      const percent = (value / max) * 100;
      slider.style.background = `linear-gradient(to right, #D9D9D9 ${percent}%, #111315 ${percent}%)`;
      tooltip.textContent = value;
      const sliderWidth = slider.offsetWidth;
      const thumbWidth = 16;
      const offset = (value / max) * (sliderWidth - thumbWidth) + thumbWidth / 2;
      tooltip.style.left = `${offset}px`;
    }
    slider.addEventListener("input", updateSlider);
    window.addEventListener("resize", updateSlider);
    updateSlider();
  });
});
// range slider end
