{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

   {% block attributes %}
   {% set filtered_attributes = [] %}
   {% for item in attributes %}
      {%- if not item.startswith('_') %}
      {% set _ = filtered_attributes.append(item) %}
      {%- endif -%}
   {%- endfor %}
   {% if filtered_attributes %}
   .. rubric:: Module attributes

   .. autosummary::
      :toctree:
   {% for item in filtered_attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% set filtered_functions = [] %}
   {% for item in functions %}
      {%- if not item.startswith('_') %}
      {% set _ = filtered_functions.append(item) %}
      {%- endif -%}
   {%- endfor %}
   {% if filtered_functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :toctree:
      :nosignatures:
   {% for item in filtered_functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% set filtered_classes = [] %}
   {% for item in classes %}
      {%- if not item.startswith('_') %}
      {% set _ = filtered_classes.append(item) %}
      {%- endif -%}
   {%- endfor %}
   {% if filtered_classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :toctree:
      :template: class.rst
      :nosignatures:
   {% for item in filtered_classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% set filtered_exceptions = [] %}
   {% for item in exceptions %}
      {%- if not item.startswith('_') %}
      {% set _ = filtered_exceptions.append(item) %}
      {%- endif -%}
   {%- endfor %}
   {% if filtered_exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
      :toctree:
   {% for item in filtered_exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% block modules %}
{% set filtered_modules = [] %}
{% for item in modules %}
   {%- if not item.startswith('_') and item != 'main' %}
   {% set _ = filtered_modules.append(item) %}
   {%- endif -%}
{%- endfor %}
{% if filtered_modules %}
.. rubric:: Submodules

.. autosummary::
   :toctree:
   :template: module.rst
   :recursive:
{% for item in filtered_modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
