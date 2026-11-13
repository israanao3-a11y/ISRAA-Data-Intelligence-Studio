/* ============================================================
   ISRAA - Data Intelligence Studio  |  Main JavaScript
   ============================================================ */

// Initialize DataTables on any table with class 'data-table'
$(document).ready(function () {
    $('.data-table').DataTable({
        pageLength: 15,
        responsive: true,
        language: {
            search: '<i class="fas fa-search"></i>',
            searchPlaceholder: 'Search...'
        }
    });
});

// ---------- Loading overlay ----------
function showLoading(message) {
    message = message || 'Processing...';
    let overlay = document.getElementById('loading-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.className = 'loading-overlay';
        overlay.innerHTML = `
            <div class="spinner-border text-primary" role="status"></div>
            <p class="mt-3 fw-semibold text-muted loading-message">${message}</p>
        `;
        document.body.appendChild(overlay);
    } else {
        overlay.querySelector('.loading-message').textContent = message;
    }
    overlay.classList.add('active');
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) overlay.classList.remove('active');
}

// ---------- Drag-and-drop upload ----------
function setupUploadArea(areaId, inputId) {
    const area = document.getElementById(areaId);
    const input = document.getElementById(inputId);
    if (!area || !input) return;

    area.addEventListener('click', () => input.click());

    area.addEventListener('dragover', (e) => {
        e.preventDefault();
        area.classList.add('drag-over');
    });

    area.addEventListener('dragleave', () => {
        area.classList.remove('drag-over');
    });

    area.addEventListener('drop', (e) => {
        e.preventDefault();
        area.classList.remove('drag-over');
        if (e.dataTransfer.files.length) {
            input.files = e.dataTransfer.files;
            input.dispatchEvent(new Event('change'));
        }
    });
}

// ---------- Flash message auto-dismiss ----------
setTimeout(function () {
    document.querySelectorAll('.alert-dismissible').forEach(function (el) {
        let bsAlert = bootstrap.Alert.getOrCreateInstance(el);
        bsAlert.close();
    });
}, 6000);

// ---------- Confirm before destructive action ----------
function confirmAction(message) {
    return confirm(message || 'Are you sure you want to proceed?');
}

// ---------- Format numbers ----------
function formatNumber(num, decimals) {
    decimals = decimals !== undefined ? decimals : 2;
    if (num === null || num === undefined || isNaN(num)) return '—';
    return Number(num).toLocaleString(undefined, {
        minimumFractionDigits: 0,
        maximumFractionDigits: decimals
    });
}
