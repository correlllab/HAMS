import omni.ext
import omni.ui as ui

class MyExtension(omni.ext.IExt):
    """This extension manages a simple counter UI."""
    def on_startup(self, _ext_id):
        print("cl_reset_button Extension startup")

        self._window = ui.Window(
            "cl_reset_button", width=100, height=100
        )
        with self._window.frame:
            with ui.VStack():

                def on_click():
                    #reset_sim()
                    #self._count += 1
                    #label.text = f"count: {self._count}"
                    print("RESETING SIM")

                #def on_reset():
                #    self._count = 0
                #    label.text = "empty"

                #on_reset()

                with ui.HStack():
                    ui.Button("Reset", clicked_fn=on_click)
                    #ui.Button("Reset", clicked_fn=on_reset)

    def on_shutdown(self):
        """This is called every time the extension is deactivated. It is used
        to clean up the extension state."""
        print("cl_reset_button Extension shutdown")
