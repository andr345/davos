#!/usr/bin/env python3

import sys
from pathlib import Path
from subprocess import Popen, PIPE
import zipapp
import tempfile

project_root = [str(p.parent) for p in Path(__file__).absolute().parents if p.name == 'davos'][0]
sys.path.append(project_root)

from davos.sys.gitignore_parser import parse_gitignore


class DeploymentTool:

    def __init__(self, target):
        self.target = target

    @staticmethod
    def run(*args, **kwargs):
        return Popen(args=[*args], stdin=PIPE, stdout=PIPE, stderr=PIPE, **kwargs)

    def upload_project(self, src_path, main, dst_path):

        src_path = Path(src_path).resolve()
        main_path = Path(main).resolve()

        ignore_file = src_path / ".deployignore"

        if ignore_file.exists():
            matches = parse_gitignore(ignore_file, base_dir=src_path)
            def zip_filter(f: Path):
                r = not matches(str(src_path / f))
                return r
        else:
            zip_filter = None

        if Path(main).exists():
            # File name to module name, with entry point in main()
            # i.e /path/to/ltr/module/start_module.py -> davos.module.start_module:main
            main = ".".join(main_path.relative_to(src_path).with_suffix("").parts) + ":main"

        mod, sep, fn = main.partition(':')
        mod_ok = all(part.isidentifier() for part in mod.split('.'))
        fn_ok = all(part.isidentifier() for part in fn.split('.'))

        if not (sep == ':' and mod_ok and fn_ok):
            err = "Cannot upload project, invalid entry point: " + main
            return "", err

        file = "%s/%s.pyz" % (dst_path, mod)
        print("Uploading %s to %s" % (src_path, file))

        with tempfile.SpooledTemporaryFile(max_size=100 * 1024 * 1024) as f:
            print(f"Packing contents of {src_path}")
            zipapp.create_archive(src_path, f, main=main, filter=zip_filter, interpreter="/usr/bin/env python3")
            f.seek(0)
            out, err = self.run("ssh", self.target, f"mkdir -p {dst_path} && touch {file} && chmod u+x {file} && cat > {file}").communicate(f.read())
            out, err = out.decode('utf-8').strip(), err.decode('utf-8').strip()

        return out, err


if __name__ == '__main__':

    import argparse

    # Example: ./deploy.py .. eval/l2dub.py <my_remote_address>  (Note: root must point to the parent, i.e top-level davos/ not to davos/davos/)
    # Will upload the entire code tree as an executable file named davos.eval.l2dub.py

    parser = argparse.ArgumentParser(description='Package a project as a ZipApp and upload it to a remote server with ssh.')
    parser.add_argument('root', type=str, help="Project root path. " +
                                               "Project files matching patterns in a file named .deployignore in this directory, will be ignored. " +
                                               ".deployignore has the same format as .gitignore.")
    parser.add_argument('main', type=str, help="Main file path: Name of the python file containing a main() function, or a zipapp.create_archive() main entry point")
    parser.add_argument('target', type=str, help="Target user@hostname:remote_path where user@hostname can be a shorthand name as specified in .ssh/config")
    args = parser.parse_args()

    # If no remote path is specified, default to "."
    remote, dst_path = (args.target, ".") if ":" not in args.target else args.target.rsplit(":", maxsplit=1)
    dt = DeploymentTool(remote)
    dt.upload_project(args.root, args.main, dst_path)
