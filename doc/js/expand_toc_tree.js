function expandTocTree() {
    var toctree = document.querySelector('button.toctree-expand');
    var parentLink = toctree.closest('a');
    var parentListItem = parentLink.closest('li');

    if (parentLink && parentLink.textContent.includes('ODE-toolbox') && (parentListItem.getAttribute("aria-expanded") === "false" || !parentListItem.hasAttribute("aria-expanded"))) {
        toctree.focus();
        toctree.click();
    }
}

function hideDeeperTocTreeItems() {
    var toctree = document.getElementsByClassName('local-toc')[0];
    const subLists = toctree.querySelectorAll('ul ul ul');
    subLists.forEach(ul => {
        ul.style.display = 'none';
    });
}

setInterval(expandTocTree, 100);
setInterval(hideDeeperTocTreeItems, 100);
