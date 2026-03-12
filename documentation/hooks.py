import os
import shutil


def on_post_build(config, **kwargs):
  """Copy extra assets to the site build directory."""
  base_dir = os.path.dirname(os.path.dirname(config['config_file_path']))
  docs_dir = config['docs_dir']
  site_dir = config['site_dir']

  extra_files = [
      (os.path.join(base_dir, 'llms.txt'), os.path.join(site_dir, 'llms.txt')),
      (
          os.path.join(docs_dir, 'producer_consumer.svg'),
          os.path.join(site_dir, 'producer_consumer.svg'),
      ),
  ]
  for src, dest in extra_files:
    if os.path.exists(src):
      shutil.copyfile(src, dest)
