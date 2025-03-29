"""Tests for Human()."""

from pybullet_helpers.robots.human import Human


def test_human(physics_client_id):
    """Tests for Human()."""

    # TODO remove
    # from pybullet_helpers.gui import create_gui_connection
    # physics_client_id = create_gui_connection()

    human = Human(physics_client_id)
