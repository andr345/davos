import sys
from pathlib import Path
project_root = [str(p.parent) for p in Path(__file__).resolve().parents if p.name == 'davos'][0]
sys.path.append(project_root)


from davos.eval.lib.analysis import evaluate_vos, zip_dataset_sequences
from davos.config import env_settings

# Package results into one file. If ground-truth exists (i.e for DAVIS 2017), the scores will be computed and stored inside.

def main():

    env = env_settings()
    results_root = env.segmentation_path
    zip_output_root = env.all_results_path / "zip_files"
    zip_output_root.mkdir(exist_ok=True)

    for eval_name in [
        'l2dub'  # 'lwl_stage2_published', 'l2du', 'l2dub', 'l2du_hard_loss'
    ]:
        # 'dv2017_val', 'dv2017_test_dev', 'yt2018_valid_all'
        for dset_name in ('dv2017_val', ):

            print(f"Working on {eval_name} on {dset_name}")

            try:
                if dset_name in ('dv2017_val',):
                    print(f" -> Evaluating results")
                    evaluate_vos(results_root / eval_name, dset_name, quiet=True)

                dst = zip_output_root / f"{eval_name}_{dset_name}.zip"
                print(f" -> Saving to {dst}")
                zip_dataset_sequences(dset_name, results_root / eval_name, dst, quiet=True, dry_run=False, skip_ytvos_all_frames=True)

            except Exception as e:
                print(f" => failed: {e}")

    print("Done")


if __name__ == '__main__':
    main()
