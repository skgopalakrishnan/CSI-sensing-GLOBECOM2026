from typing import Dict, Tuple, Optional


def print_results_table(results: Dict[Tuple[str, str], Tuple[float, float]], results_name: str = "Clean",
                        dataset: Optional[list] = None, models: Optional[list] = None) -> None:
    # Build table headers
    datasets = dataset if dataset is not None else ['NTU-Fi-HumanID', 'NTU-Fi_HAR']
    models = models if models is not None else ['LeNet', 'ResNet18', 'BiLSTM']
    col_width = 14
    header = 'Model'.ljust(col_width) + ''.join(ds.ljust(col_width) for ds in datasets)
    print(f'\n{results_name} Accuracies (Accuracy%)')
    print('-' * len(header))
    print(header)
    print('-' * len(header))
    # Print each row
    for m in models:
        row = m.ljust(col_width)
        for ds in datasets:
            acc_loss = results.get((ds, m))
            if acc_loss is None:
                cell = 'N/A'
            else:
                acc = acc_loss[0] * 100.0
                cell = f"{acc:.2f}"
            row += cell.ljust(col_width)
        print(row)
    print('-' * len(header))
    return None
