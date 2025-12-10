"""
Copyright (c) 2025 Samsung Electronics Co., Ltd.

Author(s):
Mahmoud Afifi (m.afifi1@samsung.com, m.3afifi@gmail.com)

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

Launch script for the photo-editing tool.
"""



import argparse
import tkinter as tk
from photo_editor import PhotoEditorUI

def get_args():
  parser = argparse.ArgumentParser(description='Photo-editing tool.')
  parser.add_argument('--full-screen', dest='full_screen', action='store_true',
                      help='Launch the tool in full-screen mode.')
  return parser.parse_args()


if __name__ == '__main__':
  args = get_args()
  root = tk.Tk()
  app = PhotoEditorUI(root, full_screen=args.full_screen)
  root.mainloop()
