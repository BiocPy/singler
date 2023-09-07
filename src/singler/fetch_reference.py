import urllib.request as req
import urllib.parse
import summarizedexperiment
import tempfile
import os
import gzip
import biocframe
import numpy


session_dir = None

def fetch_reference(name: str, cache_dir: str = None):
    name_choices = set([ 
        "BlueprintEncode", 
        "DatabaseImmuneCellExpression", 
        "HumanPrimaryCellAtlas", 
        "MonacoImmune", 
        "NovershternHematopoietic", 
        "ImmGen", 
        "MouseRNAseq"
    ])

    if name not in name_choices:
        raise ValueError("'" + name + "' is not a recognized reference dataset")

    all_files = { "matrix": name + "_matrix.csv.gz" }
    gene_types = [ "ensembl", "entrez", "symbol" ]
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

    base_url = "https://github.com/kanaverse/singlepp-references/releases/download/2023-04-28"

    if cache_dir is None:
        # This should already lie inside the OS's temporary directory, based on
        # documentation for tempfile.gettempdir(); no need to clean it up afterwards.
        if session_dir is None:
            session_dir = tempfile.mkdtemp()
        cache_dir = session_dir
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
        with gzip.open(all_paths["markers_" + lab], "rt") as handle:
            for line in handle:
                fields = line.strip().split("\t")

                first = all_label_names[int(fields[0])]
                if first not in current_markers:
                    current_markers[first] = {}
                inner = current_markers[first]

                second = all_label_names[int(fields[1])]
                inner[second] = [int(j) for j in fields[2:]]

        markers[lab] = current_markers

    # Reading in genes.
    gene_ids = {} 
    for g in gene_types:
        with gzip.open(all_paths["genes_" + g], "rt") as handle:
            current_genes = []
            for line in handle:
                current_genes.append(line.strip())
            gene_ids[g] = current_genes

    row_data = biocframe.BiocFrame(gene_ids)
    col_data = biocframe.BiocFrame(labels)

    # Reading in the matrix first.
    mat = numpy.ndarray((row_data.shape[0], col_data.shape[0]), dtype=numpy.int32, order="F")
    with gzip.open(all_paths["matrix"], "rt") as handle:
        sample = 0
        for line in handle:
            contents = line.strip().split(",")
            for i, x in enumerate(contents):
                contents[i] = int(x)
            mat[:,sample] = contents
            sample += 1

    return summarizedexperiment.SummarizedExperiment(
        { "ranks": mat }, 
        row_data = row_data, 
        col_data = col_data, 
        metadata = markers
    )
