# MINDIGENOUS Supabase Setup

The app expects these public tables in Supabase:

- `profiles`
- `projects`
- `billing_customers`

Apply `supabase/schema.sql` in the Supabase SQL editor, or run the migration in
`supabase/migrations/20260508143000_cloud_workspace_data.sql` after linking the
project with the Supabase CLI.

The browser app only uses the anon key. It cannot create or migrate database
tables from the client, so schema setup must be done with dashboard SQL access,
a service role environment, or the Supabase CLI.
