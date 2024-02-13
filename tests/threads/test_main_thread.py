from ami.threads.main_thread import MainThread, ThreadTypes
from ami.threads.shared_object_pool import SharedObjectNames, SharedObjectPool


class TestMainThread:
    def test_object_sharing(self):
        shared_object_pool = SharedObjectPool()
        main_thread = MainThread()
        main_thread.attach_shared_object_pool(shared_object_pool)

        # test can get.
        shared_object_pool.get(ThreadTypes.MAIN, SharedObjectNames.THREAD_COMMAND_HANDLERS)
