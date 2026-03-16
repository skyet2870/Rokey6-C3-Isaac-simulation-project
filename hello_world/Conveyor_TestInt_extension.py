import os

import omni.ext
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate

from .ConveyorTestInt import ConveyorTestInt


class ConveyorIntExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.example_name = "ConveyorTestInt"
        self.category = "rokey"

        ui_handle = BaseSampleUITemplate(
            ext_id=ext_id,
            file_path=os.path.abspath(__file__),
            title="ConveyorTestInt",
            sample=ConveyorTestInt(),
        )

        get_browser_instance().register_example(
            name=self.example_name,
            execute_entrypoint=ui_handle.build_window,
            ui_hook=ui_handle.build_ui,
            category=self.category,
        )

    def on_shutdown(self):
        get_browser_instance().deregister_example(
            name=self.example_name,
            category=self.category,
        )