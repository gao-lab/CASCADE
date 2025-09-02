{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

   {% block methods %}
   {% set filtered_methods = [] %}
   {% for item in methods %}
      {%- if item in members and item not in inherited_members and not item.startswith('_') %}
      {% set _ = filtered_methods.append(item) %}
      {%- endif -%}
   {%- endfor %}
   {% if filtered_methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree:
      :nosignatures:
   {% for item in filtered_methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% set filtered_attributes = [] %}
   {% for item in attributes %}
      {%- if item in members and item not in inherited_members and not item.startswith('_') %}
      {% set _ = filtered_attributes.append(item) %}
      {%- endif -%}
   {%- endfor %}
   {% if filtered_attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in filtered_attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
