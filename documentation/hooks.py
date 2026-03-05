import os
import shutil


def on_post_build(config, **kwargs):
  """Copy llms.txt to the site build directory."""
  base_dir = os.path.dirname(os.path.dirname(config['config_file_path']))
  src = os.path.join(base_dir, 'llms.txt')
  dest = os.path.join(config['site_dir'], 'llms.txt')
  if os.path.exists(src):
    shutil.copyfile(src, dest)
