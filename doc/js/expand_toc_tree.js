function expandTocTree() {
    var toctree = document.querySelector('button.toctree-expand');
    var parentLink = toctree.closest('a');
    var parentListItem = parentLink.closest('li');

    if (parentLink && parentLink.textContent.includes('ODE-toolbox') && (parentListItem.getAttribute("aria-expanded") === "false" || !parentListItem.hasAttribute("aria-expanded"))) {
        toctree.focus();
        toctree.click();
    }
}

function expandTocTreeTimer() {
    setInterval(expandTocTree, 100);
}

expandTocTreeTimer();
