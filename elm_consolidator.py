from data_loader import get_smith_labels, create_single_hdf_file

smith_shots = get_smith_labels()
create_single_hdf_file(dest='smith_traces.h5', smith_shots=smith_shots)

