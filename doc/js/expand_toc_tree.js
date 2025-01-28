function expandTocTree() {
    var toctree = document.querySelector('button.toctree-expand');

    toctree.focus ();
    toctree.click ();
}

function expandTocTreeTimer() {                        
    setTimeout (expandTocTree);
}

function init() {
    var anchors = document.querySelectorAll('a.reference.internal');
    expandTocTree();
    for (var anchor of anchors) {
        anchor.addEventListener ('click', expandTocTreeTimer);
    }
}

window.addEventListener ('load', init);
