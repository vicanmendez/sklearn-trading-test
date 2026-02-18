// Theme Toggler
document.addEventListener('DOMContentLoaded', () => {
    const toggleBtn = document.getElementById('theme-toggle');
    const body = document.body;
    const icon = toggleBtn.querySelector('i');

    // Check saved theme
    const currentTheme = localStorage.getItem('theme');
    if (currentTheme === 'light') {
        body.setAttribute('data-theme', 'light');
        icon.classList.remove('fa-sun');
        icon.classList.add('fa-moon');
        toggleBtn.classList.replace('btn-outline-warning', 'btn-outline-dark');
    }

    toggleBtn.addEventListener('click', () => {
        if (body.hasAttribute('data-theme')) {
            body.removeAttribute('data-theme');
            localStorage.setItem('theme', 'dark');
            icon.classList.remove('fa-moon');
            icon.classList.add('fa-sun');
            toggleBtn.classList.replace('btn-outline-dark', 'btn-outline-warning');
        } else {
            body.setAttribute('data-theme', 'light');
            localStorage.setItem('theme', 'light');
            icon.classList.remove('fa-sun');
            icon.classList.add('fa-moon');
            toggleBtn.classList.replace('btn-outline-warning', 'btn-outline-dark');
        }
    });
});

// Bot Control (for Dashboard)
function controlBot(botId, action) {
    if (!confirm(`Are you sure you want to ${action} this bot?`)) return;

    fetch(`/api/bot/${botId}/${action}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
        .then(response => response.json())
        .then(data => {
            alert(data.message);
            location.reload();
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred.');
        });
}
