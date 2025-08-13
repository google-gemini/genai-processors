import asyncio
import unittest

from absl.testing import absltest
from genai_processors import context


class ContextTest(unittest.TestCase):

  def test_create_task_different_context(self):
    """Tests that tasks created in different contexts use different task groups."""

    async def create_and_get_task_group():
      """Creates a task group and returns it."""
      async with context.context() as task_group:
        task = context.create_task(asyncio.sleep(0.1))
        await task
        return task_group

    async def test():
      return await asyncio.gather(
          create_and_get_task_group(),
          create_and_get_task_group(),
      )

    # Create two tasks in different contexts
    task_group1, task_group2 = asyncio.run(test())

    # Ensure the task groups are different
    self.assertIsNot(task_group1, task_group2)

  def test_lookup_returns_none(self):
    """Tests that contexts can be nested."""

    async def test():
      self.assertIsNone(context.task_group())
      async with context.context():
        assert isinstance(context.task_group(), asyncio.TaskGroup)

    asyncio.run(test())

  def test_create_task_nested_context(self):
    """Tests that contexts can be nested."""

    async def test():
      async with context.context():
        parent = context.task_group()
        async with context.context():
          child = context.task_group()
          assert parent is not child
        assert context.task_group() is parent

    asyncio.run(test())

  def test_child_tasks_inherit_the_context(self):

    async def child():
      return context.task_group()

    async def test():
      async with context.context():
        child_task = context.create_task(child())
        tg = await child_task
        assert tg is context.task_group()

    asyncio.run(test())

  def test_cancel(self):
    """Tests that cancelling a context cancels all tasks in the context."""

    async def test():
      async with context.context() as ctx:
        t = context.create_task(asyncio.sleep(0.1))
        ctx.cancel()
        assert t.cancelling()

    asyncio.run(test())


if __name__ == '__main__':
  absltest.main()
