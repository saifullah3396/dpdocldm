class RegistryGroupNotFoundError(Exception):
    """
    Exception raised when a requested registry is not found.
    This exception is intended to be used within the context of the registry
    system to indicate that an attempt to access a non-existent registry has
    been made.
    Attributes:
        None
    """


class RegistrySubgroupNotFoundError(Exception):
    """
    Exception raised when a specified registry subgroup is not found.

    This exception is intended to be used within the registry module to indicate
    that an operation has failed due to the absence of a required subgroup.

    Attributes:
        None
    """


class ModuleNotFoundError(Exception):
    """
    Exception raised when a specified module is not found in the registry.
    """
