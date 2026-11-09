from itertools import repeat

from earthkit.utils.parameters import PARAMS


def _field_ufunc(func, *args, fieldlist_ufunc_kwargs=None, **kwargs):
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
        This can include 'parameters', 'param_ids', 'default_parameter' and 'parameter_definitions'

        - 'parameters': dict, optional
            A mapping of input field parameter.variable values to output parameter variable
            names. The output parameters names must be defined in the parameter database
            (``_PARAMS``) or in 'parameter_definitions'.
        - 'param_ids': dict, optional
            A mapping of input metadata.paramId (the ecCodes GRIB parameter ID) values to output parameter
            variable names.
            The output parameters names must be defined in the parameter database (``_PARAMS``)
            or in 'parameter_definitions'.
        - 'default_parameter': str, optional
            The default parameter variable name to use if no mapping is found in
            'parameters' or 'param_ids'. This must be defined in the parameter database
            (``_PARAMS``) or in 'parameter_definitions'.
        - 'parameter_definitions': dict, optional
            Parameter definitions supplied by the caller, with the same shape as the parameter
            database (``_PARAMS``). They are looked up before the built-in definitions, so a
            definition with the same name overrides the built-in one, for this call only.
            This allows callers in other earthkit packages (e.g. earthkit-meteo) to work with
            parameters which are not yet in earthkit-utils; the final goal is to store all
            the parameters in earthkit-utils.

        The algorithm for determining the output parameter variable name is as follows:
        1. If 'parameters' is provided, check if the first input field's parameter.variable
            is in the mapping. If so, use the corresponding output variable name.
        2. If 'param_ids' is provided, check if the first input field's metadata.paramId
            is in the mapping. If so, use the corresponding output variable name.
        3. If neither mapping yields a result, use 'default_parameter' if provided.
        4. If no output parameter variable name can be determined, raise a ValueError.

        Once the output parameter variable name is determined, the corresponding metadata
        (parameter.variable and parameter.units) will be looked up in the parameter database
        (``_PARAMS``), or in 'parameter_definitions', and set on the resulting Field.
    **kwargs: dict
        Additional keyword arguments to pass to the function.
    """
    import earthkit.data as ekd

    fields = args
    field = fields[0]
    assert isinstance(field, ekd.Field), "field_ufunc first argument must be a Field"
    v = func(*(field.values if isinstance(field, ekd.Field) else field for field in fields), **kwargs)

    if fieldlist_ufunc_kwargs is None:
        fieldlist_ufunc_kwargs = {}

    # determine the metadata to set on the resulting Field
    parameters = fieldlist_ufunc_kwargs.get("parameters", {})
    param_ids = fieldlist_ufunc_kwargs.get("param_ids", {})
    default_parameter = fieldlist_ufunc_kwargs.get("default_parameter")

    name = None
    if parameters:
        var_in = field.get("parameter.variable", default=None)
        if var_in is not None:
            name = parameters.get(var_in)

    if name is None and param_ids:
        param_id_in = field.get("metadata.paramId", default=None)
        if param_id_in is not None:
            name = param_ids.get(param_id_in)

    if name is None:
        name = default_parameter

    if name is None:
        raise ValueError(
            "Could not determine parameter name for the resulting Field. Please provide "
            "a 'default_parameter' in 'fieldlist_ufunc_kwargs'."
        )

    # Definitions supplied by the caller, with the same shape as the parameter database (_PARAMS).
    # They are looked up before the built-in ones, which allows callers in other earthkit packages
    # (e.g. earthkit-meteo) to use parameters which are not yet in earthkit-utils. The final goal
    # is to store all the parameters in earthkit-utils.
    parameter_definitions = fieldlist_ufunc_kwargs.get("parameter_definitions")

    param_item = PARAMS.get(name, definitions=parameter_definitions)

    if param_item is None:
        raise ValueError(f"Unknown parameter '{name}' specified in fieldlist_ufunc_kwargs")
    parameter_kwargs = {"parameter.variable": param_item["variable"], "parameter.units": param_item["units"]}

    metadata = {"values": v, **parameter_kwargs}
    vertical = fieldlist_ufunc_kwargs.get("vertical")
    if vertical is not None:
        metadata["vertical"] = vertical
    result = field.set(metadata)

    return result


def fieldlist_ufunc(func, *args, fieldlist_ufunc_kwargs=None, **kwargs):
    """Apply a function to the values of earthkit.data Field or FieldList objects.

    Each field of the input FieldList objects is passed to ``func`` together with the
    fields at the same position in the other arguments, and the resulting values are
    stored in a new field. The parameter metadata (parameter.variable and parameter.units)
    of the new fields is determined from ``fieldlist_ufunc_kwargs``.

    Parameters
    ----------
    func: callable
        The function to apply to the values of the fields.
    *args: tuple
        The Field or FieldList objects to which the function will be applied. When more
        than one is given the fields are consumed in parallel.
    fieldlist_ufunc_kwargs: dict, optional
        Options controlling how the resulting fields are built, such as 'parameters',
        'param_ids', 'default_parameter' and 'parameter_definitions'. See ``_field_ufunc``
        for the full list of options and their meaning. The 'parameter_definitions' entries
        are checked upfront, even when they are not used.
    **kwargs: dict
        Additional keyword arguments to pass to ``func``.

    Returns
    -------
    :class:`earthkit.data.Field` or :class:`earthkit.data.FieldList`
        A Field when the arguments are Fields, otherwise a FieldList with one field per
        field of the input FieldList objects.
    """
    import earthkit.data as ekd

    if fieldlist_ufunc_kwargs is None:
        fieldlist_ufunc_kwargs = {}

    # check the caller supplied definitions once, even when they end up not being used, e.g.
    # when the input is an empty FieldList
    PARAMS.check_definitions(fieldlist_ufunc_kwargs.get("parameter_definitions"))

    if args:
        if isinstance(args[0], ekd.Field):
            return _field_ufunc(func, *args, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs, **kwargs)
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
                fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs,
                **kwargs,
            )
        )

    return ekd.FieldList.from_fields(result)
