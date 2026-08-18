from itertools import repeat

from earthkit.utils.parameters import FIELD_PARAMS


def _field_ufunc(func, *args, fieldlist_ufunc_kwargs={}, **kwargs):
    """
    Apply a function to the values of earthkit.data Field or FieldList objects.

    Parameters
    ----------
    func: callable
        The function to apply to the values of the Field or FieldList objects.
    *args: tuple
        The Field or FieldList objects to which the function will be applied.
    fieldlist_ufunc_kwargs: dict, optional
        A dictionary of keyword arguments to pass to the function when applied to FieldList objects.
        This can include 'variables', 'param_ids', 'default_variable'

        - 'variables': dict, optional
            A mapping of input field parameter.variable values to output parameter variable
            names. The output parameters names must be  defined in FIELD_PARAMS.
        - 'param_ids': dict, optional
            A mapping of input metadata.paramId values to output parameter variable names.
            The output parameters names must be defined in FIELD_PARAMS.
        - 'default_variable': str, optional
            The default parameter variable name to use if no mapping is found in
            'variables' or 'param_ids'. This must be defined in FIELD_PARAMS.

        The algorithm for determining the output parameter variable name is as follows:
        1. If 'variables' is provided, check if the first input field's parameter.variable
            is in the mapping. If so, use the corresponding output variable name.
        2. If 'param_ids' is provided, check if the first input field's metadata.paramId
            is in the mapping. If so, use the corresponding output variable name.
        3. If neither mapping yields a result, use 'default_variable' if provided.
        4. If no output parameter variable name can be determined, raise a ValueError.

        Once the output parameter variable name is determined, the corresponding metadata
        (parameter.variable and parameter.units) will be looked up in FIELD_PARAMS and set
        on the resulting Field.
    **kwargs: dict
        Additional keyword arguments to pass to the function.
    """
    import earthkit.data as ekd

    fields = args
    field = fields[0]
    assert isinstance(field, ekd.Field), "field_ufunc first argument must be a Field"
    v = func(*(field.values if isinstance(field, ekd.Field) else field for field in fields), **kwargs)

    # determine the metadata to set on the resulting Field
    variables = fieldlist_ufunc_kwargs.get("variables", {})
    param_ids = fieldlist_ufunc_kwargs.get("param_ids", {})
    default = fieldlist_ufunc_kwargs.get("default_variable")

    name = None
    if variables:
        var_in = field.get("parameter.variable", default=None)
        if var_in is not None:
            name = variables.get(var_in)

    if name is None and param_ids:
        param_id_in = field.get("metadata.paramId", default=None)
        if param_id_in is not None:
            name = param_ids.get(param_id_in)

    if name is None:
        name = default

    if name is None:
        raise ValueError(
            "Could not determine parameter name for the resulting Field. Please provide "
            "a 'default_variable' in 'fieldlist_ufunc_kwargs'."
        )

    param_item = FIELD_PARAMS.get(name)

    if param_item is None:
        raise ValueError(f"Unknown parameter '{name}' specified in fieldlist_ufunc_kwargs")
    parameter_kwargs = {"parameter.variable": param_item["variable"], "parameter.units": param_item["units"]}
    result = field.set({"values": v, **parameter_kwargs})

    return result


def fieldlist_ufunc(func, *args, fieldlist_ufunc_kwargs={}, **kwargs):
    import earthkit.data as ekd

    if args:
        if isinstance(args[0], ekd.Field):
            return _field_ufunc(func, *args, fieldlist_ufunc_kwargs, **kwargs)
        elif not (isinstance(args[0], ekd.FieldList)):
            raise TypeError(
                "fieldlist_ufunc arguments must be Field or FieldList instances. Found unsupported type: "
                + str(type(args[0]))
                + " in args"
            )
    else:
        raise ValueError("fieldlist_ufunc requires at least one argument")

    # an argument that is None is replaced with an infinite repeat of None to allow zipping without worrying
    # about lengths
    safe_args = [arg if arg is not None else repeat(None) for arg in args]

    result = []
    for fields in zip(*safe_args):
        result.append(
            _field_ufunc(
                func,
                *fields,
                fieldlist_ufunc_kwargs,
                **kwargs,
            )
        )

    return ekd.FieldList.from_fields(result)
