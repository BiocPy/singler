import urllib.request as req
import urllib.parse
import summarizedexperiment
import tempfile
import os
import gzip
import biocframe
import numpy
from typing import Literal, Any, Sequence, Optional
from .get_classic_markers import number_of_classic_markers


SESSION_DIR = None

KNOWN_REFERENCE = Literal[
    "BlueprintEncode",
    "DatabaseImmuneCellExpression",
    "HumanPrimaryCellAtlas",
    "MonacoImmune",
    "NovershternHematopoietic",
    "ImmGen",
    "MouseRNAseq",
]


def fetch_github_reference(
    name: KNOWN_REFERENCE, cache_dir: str = None
) -> summarizedexperiment.SummarizedExperiment:
    """Fetch a reference dataset from the
    `pre-compiled GitHub registry <https://github.com/kanaverse/singlepp-references>`_,
    for use in annotation with other **singler** functions.

    Args:
        name (KNOWN_REFERENCE): Name of the reference dataset.

        cache_dir (str, optional): Path to a cache directory in which to store
            the files downloaded from the remote. If the files are already
            present, the download is skipped.

    Returns:
        SummarizedExperiment: The reference dataset as a SummarizedExperiment,
        parts of which can be passed to :py:meth:`~singler.build_single_reference.build_single_reference`.

    Specifically, the ``ranks`` assay of the output can be used as ``ref`` in 
    :py:meth:`~singler.build_single_reference.build_single_reference`;
    one of the labels in the column data can be used as ``labels``;
    and one of the gene types in the row data can be used as ``features``.

    As the ranks are not log-normalized values, users should also use
    the relevant pre-computed marker list in the metadata. The selected
    marker list should match up with the chosen set of ``labels``. In
    addition, the markers are stored as row indices and need to be converted 
    to feature identifiers; this is achieved by passing the marker list to
    :py:meth:`~singler.fetch_reference.realize_github_markers` with the same
    gene types that were used in ``features``. The output can then be passed
    as ``markers`` in the `build_reference()` call.
    """

    all_files = {"matrix": name + "_matrix.csv.gz"}
    gene_types = ["ensembl", "entrez", "symbol"]
    for g in gene_types:
        suff = "genes_" + g
        all_files[suff] = name + "_" + suff + ".csv.gz"

    lab_types = ["fine", "main", "ont"]
    for lab in lab_types:
        suff = "labels_" + lab
        all_files[suff] = name + "_" + suff + ".csv.gz"
        suff = "label_names_" + lab
        all_files[suff] = name + "_" + suff + ".csv.gz"
        suff = "markers_" + lab
        all_files[suff] = name + "_" + suff + ".gmt.gz"

    base_url = (
        "https://github.com/kanaverse/singlepp-references/releases/download/2023-04-28"
    )

    if cache_dir is None:
        global SESSION_DIR
        # This should already lie inside the OS's temporary directory, based on
        # documentation for tempfile.gettempdir(); no need to clean it up afterwards.
        if SESSION_DIR is None:
            SESSION_DIR = tempfile.mkdtemp()
        cache_dir = SESSION_DIR
    elif not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    all_paths = {}
    for k, v in all_files.items():
        url = base_url + "/" + v
        path = os.path.join(cache_dir, urllib.parse.quote(url, safe=""))
        if not os.path.exists(path):
            req.urlretrieve(url=url, filename=path)
        all_paths[k] = path

    # Reading in labels.
    labels = {}
    markers = {}
    for lab in lab_types:
        all_labels = []
        with gzip.open(all_paths["labels_" + lab], "rt") as handle:
            for line in handle:
                all_labels.append(int(line.strip()))

        all_label_names = []
        with gzip.open(all_paths["label_names_" + lab], "rt") as handle:
            for line in handle:
                all_label_names.append(line.strip())

        for i, x in enumerate(all_labels):
            all_labels[i] = all_label_names[x]
        labels[lab] = all_labels

        current_markers = {}
        for x in all_label_names:
            current_inner = {}
            for x2 in all_label_names:
                current_inner[x2] = []
            current_markers[x] = current_inner

        with gzip.open(all_paths["markers_" + lab], "rt") as handle:
            for line in handle:
                fields = line.strip().split("\t")
                first = all_label_names[int(fields[0])]
                second = all_label_names[int(fields[1])]
                current_markers[first][second] = [int(j) for j in fields[2:]]

        markers[lab] = current_markers

    # Reading in genes.
    gene_ids = {}
    for g in gene_types:
        with gzip.open(all_paths["genes_" + g], "rt") as handle:
            current_genes = []
            for line in handle:
                y = line.strip()
                if y == "":
                    y = None
                current_genes.append(y)
            gene_ids[g] = current_genes

    row_data = biocframe.BiocFrame(gene_ids)
    col_data = biocframe.BiocFrame(labels)

    # Reading in the matrix first.
    mat = numpy.ndarray(
        (row_data.shape[0], col_data.shape[0]), dtype=numpy.int32, order="F"
    )
    with gzip.open(all_paths["matrix"], "rt") as handle:
        sample = 0
        for line in handle:
            contents = line.strip().split(",")
            for i, x in enumerate(contents):
                contents[i] = int(x)
            mat[:, sample] = contents
            sample += 1

    return summarizedexperiment.SummarizedExperiment(
        {"ranks": mat}, row_data=row_data, col_data=col_data, metadata=markers
    )


def realize_github_markers(markers: dict[Any, dict[Any, Sequence]], features: Sequence, number: Optional[int] = None):
    """Convert marker indices from a GitHub reference dataset into feature
    identifiers.  This allows the markers to be used in
    :py:meth:`~singler.build_single_reference.build_single_reference`.

    Args:
        markers (dict[Any, dict[Any, Sequence]]):
            Upregulated markers for each pairwise comparison between labels.
            Specifically, ``markers[a][b]`` should be a sequence of features
            that are upregulated in ``a`` compared to ``b``. Features are
            represented as indices into ``features``.

        features (Sequence):
            Sequence of identifiers for each feature. Features with no valid
            identifier for a particular gene type (e.g., no known symbol)
            should be represented by None.

        number (int, optional):
            Number of markers to retain. If None, we default to 
            :py:meth:`~singler.get_classic_markers.number_of_classic_markers`.

    Returns:
        dict[Any, dict[Any, Sequence]]: A dictionary with the same structure
        as ``markers``, where each inner sequence contains the corresponding
        feature identifiers in ``features``. Feature identifiers are guaranteed
        to be non-None and should have length ``number`` (or less, if not
        enough non-None identifiers are available).
    """
    if number is None:
        number = number_of_classic_markers(len(markers))

    output = {}
    for k, v in markers.items():
        current = {}
        for k2, v2 in v.items():
            n = min(len(v2), number)
            renamed = []
            for i in v2:
                if len(renamed) == n:
                    break
                feat = features[i]
                if feat is not None:
                    renamed.append(feat)
            current[k2] = renamed
        output[k] = current

    return output
