document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('cityForm');
    const spinner = document.getElementById('spinner');
    const progressText = document.getElementById('progressText');
    const showVenuesBtn = document.getElementById('showVenuesBtn');
    const cancelBtn = document.getElementById('cancelBtn');
    const errorMessage = document.getElementById('errorMessage');
    const userType = document.getElementById('userType');
    const venueTypeGroup = document.getElementById('venueTypeGroup');
    const aboutToolCard = document.getElementById('aboutToolCard');

    let controller = null;

    if (userType) {
        userType.addEventListener('change', function() {
            if (this.value === 'user') {
                venueTypeGroup.style.display = 'block';
                aboutToolCard.style.display = 'none';
                form.action = "/user_result";
            } else {
                venueTypeGroup.style.display = 'none';
                aboutToolCard.style.display = 'block';
                form.action = "/restaurant_owner_result";
            }
        });
    }

    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            errorMessage.style.display = 'none';
            showVenuesBtn.style.display = 'none';
            cancelBtn.style.display = 'block';
            spinner.style.display = 'block';
            progressText.textContent = 'Loading...';

            controller = new AbortController();
            const signal = controller.signal;

            fetch(form.action, {
                method: 'POST',
                body: new FormData(form),
                signal: signal
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.text();
            })
            .then(html => {
                document.body.innerHTML = html;
            })
            .catch(error => {
                if (error.name === 'AbortError') {
                    console.log('Fetch aborted');
                } else if (error.message === 'Network response was not ok') {
                    errorMessage.textContent = 'No network available, please try again after connecting to the internet.';
                } else {
                    errorMessage.textContent = 'Not enough venues';
                }
                errorMessage.style.display = 'block';
            })
            .finally(() => {
                spinner.style.display = 'none';
                showVenuesBtn.style.display = 'block';
                cancelBtn.style.display = 'none';
                controller = null;
                progressText.textContent = '';
            });
        });
    }

    if (cancelBtn) {
        cancelBtn.addEventListener('click', function() {
            if (controller) {
                controller.abort();
                spinner.style.display = 'none';
                showVenuesBtn.style.display = 'block';
                cancelBtn.style.display = 'none';
                errorMessage.style.display = 'none';
                progressText.textContent = '';
            }
        });
    }
});
