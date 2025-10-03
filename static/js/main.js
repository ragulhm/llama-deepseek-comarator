// static/js/main.js

document.addEventListener('DOMContentLoaded', function() {
    console.log('LLM Comparator v2 JavaScript Loaded!');

    // Example: Basic client-side form validation (can be expanded)
    const compareForm = document.querySelector('form[action="/compare"]');
    if (compareForm) {
        compareForm.addEventListener('submit', function(event) {
            const promptInput = document.getElementById('prompt');
            if (!promptInput.value.trim()) {
                alert('Please enter a prompt before comparing models.');
                promptInput.focus();
                event.preventDefault(); // Stop form submission
                return false;
            }

            // You could add more validation here for numeric inputs, etc.
            // For example, ensuring temperature is within a valid range.

            console.log('Form is being submitted...');
        });
    }

    // Example of a simple dynamic element (e.g., a "loading" indicator)
    // This would typically involve showing/hiding elements after form submission.
    // For now, let's just log a message.
    const submitButton = document.querySelector('button[type="submit"]');
    if (submitButton) {
        submitButton.addEventListener('click', function() {
            // In a real app, you might show a spinner here
            // submitButton.textContent = 'Comparing...';
            // submitButton.disabled = true;
            console.log('Comparison initiated, waiting for server response...');
        });
    }
});