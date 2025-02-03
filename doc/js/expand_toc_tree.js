function expandTocTree() {
    var toctree = document.querySelector('button.toctree-expand');
    var parentLink = toctree.closest('a');

    if (parentLink && parentLink.textContent.includes('ODE-toolbox')) {
        toctree.focus();
        toctree.click();
    }
}

function expandTocTreeTimer() {
    setTimeout(expandTocTree, 500);
}

expandTocTreeTimer();
